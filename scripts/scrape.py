from bs4 import BeautifulSoup
import urllib2
import requests
import sys
import time
import os

MAX_PAGES_PER_PROBLEM = 1

def get_soup(url):
    for attempt in range(10):
        try:
            response = urllib2.urlopen(url)
            html = response.read()

            soup = BeautifulSoup(html, 'html.parser')
            return soup
        except Exception as e:
            time.sleep(1 + attempt)
    response = urllib2.urlopen(url)
    html = response.read()

    soup = BeautifulSoup(html, 'lxml')
    return soup

def get_problem_statement(url):
    soup = get_soup(url)
    statement = soup.find('div', {'class': 'problem-statement'})
    return str(statement)


cookies = None
done_test = False
def get_problem_solutions(problem_id, url, lang):
    global cookies
    global done_test

    langs = ['cpp.g++17', 'cpp.g++14', 'cpp.g++11', 'cpp.g++', 'cpp.ms', 'cpp.ms2017']                       

    os.mkdir(problem_id, 0755);

    testcases = str(problem_id + "/" + problem_id + "_" + "testcases" + ".txt")
    testcases = open(testcases, "w")
    done_test = False

    for lang_ in langs:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        csrf = soup.find("meta",  {'name': "X-Csrf-Token"})['content']
        cookies = r.cookies
        r = requests.post(url, {'csrf_token': str(csrf), 'action': 'setupSubmissionFilter', 'verdictName': 'anyVerdict', 'programTypeForInvoker': lang_, 'comparisonType': 'NOT_USED', 'judgedTestCount': '', '_tta': 320, 'frameProblemIndex': url[-1]}, cookies=cookies)
        
        def extract_data(url):
            global done_test
            content = requests.get(url, cookies=cookies).text
            soup = BeautifulSoup(content, 'html.parser')
            links = soup.find_all('a', {'class': 'view-source'})
            verdicts = soup.find_all('span', {'class': 'submissionVerdictWrapper'})
            
            for link, verdict in zip(links, verdicts):
                if(verdict.get_text() == "Accepted"):
                    filename = str(problem_id + "/" + link['href'].split("/")[-1] + ".cpp")
                    filename = open(filename, "w")
                    for i in range(10):
                        sp = get_soup('http://codeforces.com/' + link['href'])
                        solution_elem = sp.find('pre', {'class': 'program-source'})
                        if solution_elem is None:
                            # print "!",
                            sys.stdout.flush()
                            time.sleep(1)
                            continue
                        # print ".",
                        break
                    else:
                        # print "#"
                        sys.stdout.flush()
                        continue
                        
                    sys.stdout.flush()
                    solution = solution_elem.get_text().rstrip()
                    if not done_test:
                        inputs = sp.find_all('div', {'class': 'input-view'})
                        outputs = sp.find_all('div', {'class': 'output-view'})
                        # answers = sp.find_all('div', {'class': 'answer-view'})
                        # gverdicts = [x for x in sp.find_all('div') if x.get_text().startswith('Verdict: ')]
                        # count = 0
                        for i, o in zip(inputs, outputs):
                            ip = i.find('pre').get_text().rstrip()
                            if(ip[-3:] == "..."):
                                continue
                            op = o.find('pre').get_text().rstrip()
                            if(op[-3:] == "..."):
                                continue
                            testcases.write(ip + "\n")
                            testcases.write("###ENDINPUT###" + "\n")
                            testcases.write(op + "\n")
                            testcases.write("###ENDOUTPUT###" + "\n")
                            # ap = a.find('pre').get_text().rstrip()
                            # vp = v.get_text()[len("Verdict: "):]
                            # print(ip)
                            # print(op)
                            # count += 1
                        # print(ap)
                        # print(vp)
                        testcases.close()   
                        done_test = True
                    filename.write(solution.encode("utf-8"))
                    filename.close()
                    # print(solution)
                    # print(count)
                    # exit(0)
            # print ""
            return soup


        soup = extract_data(url)

        num_pages = len(soup.find_all('span', {'class': 'page-index'}))
        if num_pages == 5:
            num_pages = int(soup.find_all('span', {'class': 'page-index'})[-1].get_text())
        # print "Pages: ", num_pages
        if num_pages > MAX_PAGES_PER_PROBLEM:
            num_pages = MAX_PAGES_PER_PROBLEM
        for i in range(num_pages + 1):
            if i >= 2:
                extract_data(url + '/page/%d' % i)



def get_problems(page):
    soup = get_soup('http://codeforces.com/problemset/page/%s' % page)
    problems = soup.find('table', {'class': 'problems'})
    ids_and_names = [x for x in problems.find_all('a') if x['href'].startswith('/problemset/problem') if x.get_text().strip() not in ['642A', 'Scheduler for Invokers', '775A', 'University Schedule']]
    statuses = [x for x in problems.find_all('a') if x['href'].startswith('/problemset/status')]

    assert len(ids_and_names) == len(statuses) * 2, "%s != %s, %s" % (len(ids_and_names), len(statuses), page)

    ret = []
    for i in range(len(statuses)):
        id_ = ids_and_names[i * 2].get_text().strip()
        name = ids_and_names[i * 2 + 1].get_text().strip()
        url = ids_and_names[i * 2]['href']
        status_url = statuses[i]['href']
        ret.append((id_, name, url, status_url))

    return ret

def crawl(start_page, last_page):
    for page in range(start_page, last_page + 1):
        problems = get_problems(page)
        for problem in problems:
            lang = 'c++'
            suffix = lang

            print "FETCHING %s %s" % (problem[0], lang)
            statement = get_problem_statement('http://codeforces.com/' + problem[2])
            # print(statement)
            name = problem[1]
            problem_id = problem[0]
            get_problem_solutions(problem_id, 'http://codeforces.com/' + problem[3], lang)
            print "Finalized"

if __name__ == "__main__":
    #get_problems(1)
    #get_problem_statement('http://codeforces.com/problemset/problem/3/B')
    #get_problem_solutions(100, 'http://codeforces.com/problemset/status/807/problem/B')
    # get_problem_solutions(100, 'http://codeforces.com/problemset/status/807/problem/B', 'c++')
    assert len(sys.argv) > 2
    crawl(int(sys.argv[1]), int(sys.argv[2]))
    
