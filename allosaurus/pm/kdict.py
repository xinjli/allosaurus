import numpy
import struct
import functools
import os.path
import gzip
import bz2
from pathlib import Path


class KaldiWriter:

    def __init__(self, path=None, scp=True):
        """
        writer of BOTH ark and scp

        :param path:
        :param scp:
        """

        self.scp = scp

        if path:
            self.open(path)

        self.ark_offset = 0

    def open(self, path):

        # remove scp or ark suffix
        path = str(path)
        if path.endswith(".scp") or path.endswith(".ark"):
            path = path[:-4]

        # mkdir
        p = Path(path).parent
        p.mkdir(parents=True, exist_ok=True)

        self.ark_path = str(path) + '.ark'
        self.scp_path = str(path) + '.scp'

        # delete old files
        if os.path.exists(self.ark_path):
            os.remove(self.ark_path)

        if os.path.exists(self.scp_path):
            os.remove(self.scp_path)

        self.ark_writer = open(self.ark_path, "ab")
        self.scp_writer = open(self.scp_path, "a")
        self.ark_offset = self.ark_writer.tell()

    def write(self, utt_ids, feats):

        # write batch
        if isinstance(utt_ids, list):
            pointers = []

            for utt_id, feat in zip(utt_ids, feats):
                self.ark_offset += write_string(self.ark_writer, utt_id)
                pointers.append("%s:%d" % (self.ark_path, self.ark_offset))
                self.ark_offset += write_matrix(self.ark_writer, feat)

            #self.ark_writer.flush()

            for utt_id, pointer in zip(utt_ids, pointers):
              self.scp_writer.write("%s %s\n" % (utt_id, pointer))

            #self.scp_writer.flush()

        else:
            # write single instance
            utt_id = utt_ids
            feat = feats
            self.ark_offset += write_string(self.ark_writer, utt_id)
            pointer = "%s:%d" % (self.ark_path, self.ark_offset)
            self.ark_offset += write_matrix(self.ark_writer, feat)
            self.scp_writer.write("%s %s\n" % (utt_id, pointer))

    def close(self):
        self.ark_writer.close()
        self.scp_writer.close()


def smart_open(filename, mode='rb', *args, **kwargs):
    '''
    Opens a file "smartly":
      * If the filename has a ".gz" or ".bz2" extension, compression is handled
        automatically;
      * If the file is to be read and does not exist, corresponding files with
        a ".gz" or ".bz2" extension will be attempted.
    '''
    readers = {'.gz': gzip.GzipFile, '.bz2': bz2.BZ2File}
    if 'r' in mode and not os.path.exists(filename):
        for ext in readers:
            if os.path.exists(filename + ext):
                filename += ext
                break
    extension = os.path.splitext(filename)[1]
    return readers.get(extension, open)(filename, mode, *args, **kwargs)


def read_string(f):
    s = ""
    while True:
        c = f.read(1).decode('utf-8')
        if c == "": raise ValueError("EOF encountered while reading a string.")
        if c == " ": return s
        s += c


def read_integer(f):
    n = ord(f.read(1))
    # return reduce(lambda x, y: x * 256 + ord(y), f.read(n)[::-1], 0)
    a = f.read(n)[::-1]
    try:
        return int.from_bytes(a, byteorder='big', signed=False)
    except:
        return functools.reduce(lambda x, y: x * 256 + ord(y), a, 0)
    # return functools.reduce(lambda x, y: x * 256 + ord(y), f.read(n)[::-1].decode('windows-1252'), 0)
    # try:
    # a=f.read(n)[::-1]
    # b=int.from_bytes(a, byteorder='big', signed=False)
    # print(a,type(a),b)
    # return functools.reduce(lambda x, y: x * 256 + ord(y), a[::-1], 0)
    # return functools.reduce(lambda x, y: x * 256 + ord(y), f.read(n)[::-1], 0)
    # except:
    #    return functools.reduce(lambda x, y: x * 256 + ord(y), f.read(n)[::-1].decode('windows-1252'), 0)


def read_compressed_matrix(fd, format):
    """ Read a compressed matrix,
        see: https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
        methods: CompressedMatrix::Read(...), CompressedMatrix::CopyToMat(...),
    """

    # Format of header 'struct',
    global_header = numpy.dtype([('minvalue', 'float32'), ('range', 'float32'), ('num_rows', 'int32'),
                                 ('num_cols', 'int32')])  # member '.format' is not written,
    per_col_header = numpy.dtype([('percentile_0', 'uint16'), ('percentile_25', 'uint16'), ('percentile_75', 'uint16'),
                                  ('percentile_100', 'uint16')])

    # Read global header,
    globmin, globrange, rows, cols = numpy.frombuffer(fd.read(16), dtype=global_header, count=1)[0]

    # The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
    #                         {           cols           }{     size         }
    col_headers = numpy.frombuffer(fd.read(cols * 8), dtype=per_col_header, count=cols)
    col_headers = numpy.array(
        [numpy.array([x for x in y]) * globrange * 1.52590218966964e-05 + globmin for y in col_headers],
        dtype=numpy.float32)
    data = numpy.reshape(numpy.frombuffer(fd.read(cols * rows), dtype='uint8', count=cols * rows),
                         newshape=(cols, rows))  # stored as col-major,

    mat = numpy.zeros((cols, rows), dtype='float32')
    p0 = col_headers[:, 0].reshape(-1, 1)
    p25 = col_headers[:, 1].reshape(-1, 1)
    p75 = col_headers[:, 2].reshape(-1, 1)
    p100 = col_headers[:, 3].reshape(-1, 1)
    mask_0_64 = (data <= 64)
    mask_193_255 = (data > 192)
    mask_65_192 = (~(mask_0_64 | mask_193_255))

    mat += (p0 + (p25 - p0) / 64. * data) * mask_0_64.astype(numpy.float32)
    mat += (p25 + (p75 - p25) / 128. * (data - 64)) * mask_65_192.astype(numpy.float32)
    mat += (p75 + (p100 - p75) / 63. * (data - 192)) * mask_193_255.astype(numpy.float32)

    return mat.T  # transpose! col-major -> row-major,


def read_compressed_matrix_shape(fd):
    """ Read a compressed matrix,
        see: https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
        methods: CompressedMatrix::Read(...), CompressedMatrix::CopyToMat(...),
    """

    # Format of header 'struct',
    global_header = numpy.dtype([('minvalue', 'float32'), ('range', 'float32'), ('num_rows', 'int32'),
                                 ('num_cols', 'int32')])  # member '.format' is not written,
    per_col_header = numpy.dtype([('percentile_0', 'uint16'), ('percentile_25', 'uint16'), ('percentile_75', 'uint16'),
                                  ('percentile_100', 'uint16')])

    # Read global header,
    globmin, globrange, rows, cols = numpy.frombuffer(fd.read(16), dtype=global_header, count=1)[0]

    return rows, cols


def read_matrix(f, dtype=numpy.float32):
    header = f.read(2).decode('utf-8')
    if header != "\0B":
        raise ValueError("Binary mode header ('\0B') not found when attempting to read a matrix.")
    format = read_string(f)

    if format == "DM":
        nRows = read_integer(f)
        nCols = read_integer(f)

        data = struct.unpack("<%dd" % (nRows * nCols), f.read(nRows * nCols * 8))
        data = numpy.array(data, dtype=dtype)
        return data.reshape(nRows, nCols)

    elif format == "FM":
        nRows = read_integer(f)
        nCols = read_integer(f)

        data = struct.unpack("<%df" % (nRows * nCols), f.read(nRows * nCols * 4))
        data = numpy.array(data, dtype=dtype)
        return data.reshape(nRows, nCols)

    elif format == "CM":
        data = read_compressed_matrix(f, format)
        return data

    else:
        raise ValueError(
            "Unknown matrix format '%s' encountered while reading; currently supported formats are DM (float64) and FM (float32)." % format)

def read_matrix_format(f):

    header = f.read(2).decode('utf-8')
    if header != "\0B":
        raise ValueError("Binary mode header ('\0B') not found when attempting to read a matrix.")

    format = read_string(f)

    return format


def read_matrix_shape(f):
    header = f.read(2).decode('utf-8')
    if header != "\0B":
        raise ValueError("Binary mode header ('\0B') not found when attempting to read a matrix.")

    format = read_string(f)

    # for compressed shape
    if format == 'CM':
        return read_compressed_matrix_shape(f)

    # for non compression shape
    nRows = read_integer(f)
    nCols = read_integer(f)

    if format == "DM":
        f.seek(nRows * nCols * 8, os.SEEK_CUR)
    elif format == "FM":
        f.seek(nRows * nCols * 4, os.SEEK_CUR)
    else:
        raise ValueError(
            "Unknown matrix format '%s' encountered while reading; currently supported formats are DM (float64) and FM (float32)." % format)
    return nRows, nCols


def write_string(f, s):
    return f.write((s + " ").encode('utf-8'))


def write_integer(f, a):
    s = struct.pack("<i", a)
    return f.write(chr(len(s)).encode('utf-8') + s)


def write_matrix(f, data):
    cnt = 0
    cnt += f.write('\0B'.encode('utf-8'))  # Binary data header
    if str(data.dtype) == "float64":
        cnt += write_string(f, "DM")
        cnt += write_integer(f, data.shape[0])
        cnt += write_integer(f, data.shape[1])
        cnt += f.write(struct.pack("<%dd" % data.size, *data.ravel()))

    elif str(data.dtype) == "float32":
        cnt += write_string(f, "FM")
        cnt += write_integer(f, data.shape[0])
        cnt += write_integer(f, data.shape[1])
        cnt += f.write(struct.pack("<%df" % data.size, *data.ravel()))
    else:
        raise ValueError(
            "Unsupported matrix format '%s' for writing; currently supported formats are float64 and float32." % str(
                data.dtype))

    return cnt

def read_matrix_by_offset(arkfile, offset, dtype=numpy.float32):
    with smart_open(arkfile, "rb") as g:
        g.seek(offset)
        feature = read_matrix(g, dtype)
    return feature


def read_scp_offset(filename, limit=numpy.inf):

    utt_ids = []
    ark_files = []
    offsets = []

    with smart_open(filename, "r") as f:
        for line in f:
            utt_id, pointer = line.strip().split()
            p = pointer.rfind(":")
            arkfile, offset = pointer[:p], int(pointer[p + 1:])
            ark_files.append(arkfile)
            offsets.append(offset)
            utt_ids.append(utt_id)

            if len(utt_ids) == limit: break
    return utt_ids, ark_files, offsets