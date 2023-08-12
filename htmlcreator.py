
data = {
    
{ "a", "ya" },

{ "b", "se" },
{ "c", "leh" },
{ "d", "ru" },
{ "e", "eh" },
{ "f", "fe" },
{ "g", "ge" },
{ "h", "he" },
{ "i", "ia" },
{ "j", "ye" },
{ "k", "lar" },
{ "l", "leh" },
{ "m", "meh" },
{ "n", "na" },
{ "o", "ohe" },
{ "p", "pa" },
{ "q", "hu" },
{ "r", "reh" },
{ "s", "see" },
{ "t", "ra" },
{ "u", "ve" },
{ "v", "va" },
{ "w", "wa" },
{ "x", "ke" },
{ "y", "yoh" },
{ "z", "sha" },

}

table_html = "<table><tr><th>Key</th><th>Value</th></tr>"
for key, value in data.items():
    table_html += f"<tr><td>{key}</td><td>{value}</td></tr>"
table_html += "</table>"

print(table_html)