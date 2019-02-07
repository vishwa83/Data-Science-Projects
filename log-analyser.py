from readability import readabilitytests

log_file = open('./final.log')

words_dict = {}
count_dict = {}


# rip the logs apart and build a list of what each nick has said
for line in log_file:
    # ignore the first 16 characters, as that's the date stamp
    #print line[10:]
    spoken_line = line[11:] if line[11:].startswith('<') else None
    if spoken_line:
        nick = spoken_line[1:spoken_line.find('>')]
        
        # this is a system message, ignore it
        if nick == "***":
            continue
        if count_dict.get(nick):
            count_dict[nick] += 1
        else:
            count_dict[nick] = 1
            
        # add a full stop, otherwise the analysis gets a bit confused
        words = spoken_line[spoken_line.find('>') +2:] +"."
        if words_dict.get(nick):
            words_dict[nick] += words
        else:
            words_dict[nick] = words
        
count_first_list = []
nick_first_list = []

# build some nice lists of how much people have said
for k in count_dict:
    nick_first_list.append((k,count_dict[k]))
    count_first_list.append((count_dict[k], k))

# sort these out so we can use them later
nick_first_list = sorted(nick_first_list)
count_first_list = sorted(count_first_list)

output_file = open('./output.html', 'w')

output_file.write("<html><head><title>#isotoma analysis</title></head><body>")

output_file.write("<table>")
for count in reversed(count_first_list):
    output_file.write("<tr>")
    output_file.write("<td>" + str(count[0]) + "</td><td><a href='#" + str(count[1]) + "'>" + count[1] +" </a></td>")
    output_file.write("</tr>")
output_file.write("</table>")

for nick in count_dict.keys():
        
    output_file.write("<p><h2><a name='" + nick + "'> " + nick + "</a></h2>")
    textengine = readabilitytests.ReadabilityTool()
    textengine.lang = "eng"

    results = textengine.getReportAll(words_dict[nick])

    output_file.write("<table>")

    for testname in textengine.analyzedVars:
        output_file.write("<tr>")
        if testname == "words":
            continue
        output_file.write("<td>" + testname + ": </td><td>" + str(textengine.analyzedVars[testname]) + "</td>")
        output_file.write("</tr>")

    output_file.write("</table><table>")
    for testname in results.keys():
         output_file.write("<td>" + testname + ": </td><td>" + str(round(results[testname](words_dict[nick]), 2)) + "</td>")
         output_file.write("</tr>")

    output_file.write("</table</p>")

output_file.write("</body></html>")
output_file.close()
