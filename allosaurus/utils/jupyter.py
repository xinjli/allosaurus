def dict2html(d, headers=None):

    html = ["<table>"]

    keys = list(d.keys())

    if headers:

        assert len(headers) >= 2

        html.append("<tr>")
        html.append("<th>{0}</th>".format(headers[0]))
        html.append("<th>{0}</th>".format(headers[1]))
        html.append("</tr>")

    if len(keys) >= 10:
        # print head 5 and tail 5
        for i in range(5):
            html.append(f"<tr><td>{keys[i]}</td><td>{str(d[keys[i]])}</td></tr>")

        html.append("<tr><td>...</td><td>...</td></tr>")

        for i in range(5):
            ri = len(keys) - i - 1
            html.append(f"<tr><td>{keys[ri]}</td><td>{str(d[keys[ri]])}</td></tr>")

    else:
        for k, v in d.items():
            html.append(f"<tr><td>{str(k)}</td><td>{str(v)}</td></tr>")

    html.append("<tr><td># {} items</td><td># {} items</td></tr>".format(len(d), len(d)))
    html.append("</table>")
    return ''.join(html)

def list2html(lst, headers=None):

    html = ["<table>"]

    if headers:

        assert len(headers) >= 1

        html.append("<tr>")
        html.append("<th>index</th>")
        html.append("<th>{0}</th>".format(headers[0]))
        html.append("</tr>")


    if len(lst) >= 10:
        # print head 5 and tail 5
        for i in range(5):
            html.append(f"<tr><td>{i}</td><td>{lst[i]}</td></tr>")

        html.append("<tr><td>...</td><td>...</td></tr>")

        for i in range(5):
            ri = len(lst)-i-1
            html.append(f"<tr><td>{ri}</td><td>{lst[ri]}</td></tr>")

    else:
        for i, item in enumerate(lst):
            html.append(f"<tr><td>{i}</td><td>{item}</td></tr>")

    html.append("<tr><td># {} items</td><td># {} items</td></tr>".format(len(d), len(d)))
    html.append("</table>")
    return ''.join(html)