Dictionary<string, string> data = new Dictionary<string, string>
{
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
};
string tableHtml = "<table><tr><th>Key</th><th>Value</th></tr>";
foreach (var kvp in data)
{
    tableHtml += $"<tr><td>{kvp.Key}</td><td>{kvp.Value}</td></tr>";
}
tableHtml += "</table>";
Console.WriteLine(tableHtml);