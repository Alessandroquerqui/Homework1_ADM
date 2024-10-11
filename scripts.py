# Say "Hello, World!" With Python
print("Hello, World!")

# Python If-Else
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
if (n%2==1):
    print("Weird")
elif (2<=n<=5):
    print("Not Weird")
elif (6<=n<=20):
    print("Weird")
else:
    print("Not Weird")

# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a+b)
print(a-b)
print(a*b)

# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a//b)
print(a/b)

# Loops
if __name__ == '__main__':
    n = int(input())
for i in range(n):
    print(i**2)

# Write a function
def is_leap(year):
    leap = False
    if year % 400 == 0:
        leap = True
    elif year % 100 == 0:
        leap= False
    elif year % 4 == 0:
        leap = True
    # Write your logic here
    
    return leap

# Print Function
if __name__ == '__main__':
    n = int(input())
for i in range (1, n+1):
    print(i , end ="")
    

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
l=[]
for i in range(x+1):
    for j in range(y+1):
        for k in range(z+1):
            if i+j+k!=n:
                l.append([i,j,k])
print(l)

# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split())) 
    maxarr = max(arr)
    lista = []
    for i in arr:
        if i != maxarr:
            lista.append(i)
    print(max(lista))

# Nested Lists
n=[]
if __name__ == '__main__':
    for _ in range(int(input())):
        name = input()
        score = float(input())
        n.append([name, score])
risultati=sorted(list(set([i[1]for i in n])))
penultimo=risultati[1]
nomi= [i[0] for i in n if i[1]==penultimo]
nomi.sort()
for name in nomi:
    print(name)


# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
scores = student_marks[query_name]
media = sum(scores) / len(scores)
mediaa = round(media, 2)
print(f"{mediaa:.2f}")

# Map and Lambda Function
cube = lambda x: x**3
def fibonacci(n):
    if n == 0:
        return []
    elif n == 1:
        return [0]
    l=[0,1]
    for i in range(n-2):
        l.append(l[-1] +l[-2])
    return l

# Validating Email Addresses With a Filter
import re
fun = lambda s: re.match(r'^[a-zA-Z0-9_-]+@[a-zA-Z0-9]+\.[a-zA-Z]{1,3}$', s) is not None

# Reduce Function

def product(fracs):
    t = reduce(lambda x, y: x * y, fracs)
    return t.numerator, t.denominator

# Lists
if __name__ == '__main__':
    N = int(input())
lista = [] 
    
for _ in range(N):
    command = input().split()  
    if command[0] == "insert":
        lista.insert(int(command[1]), int(command[2]))
    elif command[0] == "print":
        print(lista) 
    elif command[0] == "remove":
        lista.remove(int(command[1])) 
    elif command[0] == "append":
        lista.append(int(command[1]))
    elif command[0] == "sort":
        lista.sort()  
    elif command[0] == "pop":
        lista.pop() 
    elif command[0] == "reverse":
        lista.reverse() 

# sWAP cASE
def swap_case(s):
    return s.swapcase()

# String Split and Join
def split_and_join(line):
    # write your code here
    linea = line
    stringa = linea.split(" ")
    return "-".join(stringa)

# What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#
def print_full_name(first, last):
    # Write your code here
    print(f"Hello {first} {last}! You just delved into python.")

# Mutations
def mutate_string(string, position, character):
    return string[:position] + character + string[position+1:]

# Find a string
def count_substring(string, sub_string):
    return sum(string[i:].startswith(sub_string) for i in range(len(string)))

# String Validators
if __name__ == '__main__':
    s = input()
string = list(s)
primo = False
sec= False
tre= False
qua = False
cin= False
for i in string:
    primo  = primo or i.isalnum()
    sec = sec or i.isalpha()
    tre = tre or i.isdigit()
    qua = qua or i.islower()
    cin= cin or i.isupper()
print(primo)
print(sec)
print(tre)
print(qua)
print(cin)

# Text Alignment
#Replace all ______ with rjust, ljust or center. 
thickness = int(input()) #This must be an odd number
c = 'H'
#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    
#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Text Wrap

def wrap(string, max_width):
    final = textwrap.fill(string, max_width)
    return final

# Designer Door Mat
# Enter your code here. Read input from STDIN. Print output to 
N,M = list(map(int,input().split()))
h=1
for i in range(N):
    if i < N//2:
        print((".|."*h).center(M,"-")) 
        h = h+2 
    elif i == N//2: 
        print("WELCOME".center(M,"-"))
    elif i > N//2:
        h -= 2 
        print((".|."*h).center(M,"-"))

# String Formatting
def print_formatted(number):
    # your code goes here
    prova = format(number,"b")
    provaa = len(prova)
    for i in range(1,n+1):
        Oct = format(i,"o")
        Hex = format(i,"x")
        Bin = format(i,"b")
        print(str(i).rjust(provaa),str(Oct).rjust(provaa),str(Hex.upper()).         rjust(provaa),str(Bin).rjust(provaa))

# Alphabet Rangoli
def print_rangoli(size):
    # your code goes here
    order = "-".join(chr(char) for char in reversed(range(ord("a"), ord("a")+size)))
    # this part inverts the order until the nth letter
    lista = []
    for i in range(1, size*2, 2):
        line = f"{order[:i]}{order[:i-1][::-1]}"
        lista.append(f'{line:-^{size * 4 - 3}}')
        print(lista[-1]) 
    print('\n'.join(reversed(lista[:-1])))

# Capitalize!

# Complete the solve function below.
def solve(s):
    return " ".join([fullname.capitalize() for fullname in s.split(" ")])

# The Minion Game
def minion_game(string):
    # your code goes here
    Stuart = 0
    Kevin = 0
    length = len(string)
    vow = "AEIOU"
    for i in range(length):
        if string[i] in vow:
            Kevin = Kevin+ length - i
        else:
            Stuart = Stuart + length - i
    if Kevin < Stuart:
        print("Stuart", str(Stuart))
    elif Kevin > Stuart:
        print("Kevin", str(Kevin))
    else:
        print("Draw")

# Merge the Tools!
def merge_the_tools(string, k):
    # your code goes here
    leng = len(string)//k
    for i in range(leng):
        print("".join(dict.fromkeys(string[i*k:(i*k)+k])))

# Introduction to Sets
def average(array):
    # your code goes here
    nuovo = set(arr)
    average = sum(nuovo)/len(nuovo)
    return(round(average, 3))

# Symmetric Difference
# Enter your code here. Read input from STDIN. Print output to STDOUT
M = int(input())
sM = set(map(int, input().split()))
N = int(input())
sN = set(map(int, input().split()))
diff = sM.symmetric_difference(sN)
sort = sorted(diff)
for i in sort:
    print(i)

# No Idea!
# Enter your code here. Read input from STDIN. Print output to STDOUT
N,M = input().split()
array = input().split()
happiness = 0
A= set(input().split())
B= set(input().split())
for i in array:
    if i in A:
        happiness = happiness+1
    elif i in B:
        happiness =happiness-1
    else:
        happiness = happiness
print(happiness)

# Set .add()
# Enter your code here. Read input from STDIN. Print output to STDOUT
N = int(input())
stampe = set()
for i in range(N):
    stampe.add(input())
print(len(stampe))

# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
N = int(input())
for i in range(N):
    commands = input().split()
    if commands[0] == "pop":
        s.pop()
    elif commands[0] == "remove":
        s.remove(int(commands[1]))
    else:
        s.discard(int(commands[1]))
print(sum(s))

# Set .union() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
e = set(map(int, input().split())) #set subscribed to English newspaper
b = int(input())
f = set(map(int, input().split())) #set subscribed to French newspaper
print(len(e.union(f)))

# Set .intersection() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
e = set(map(int, input().split())) #like before
b = int(input())
f = set(map(int, input().split()))
print(len(e & f))

# Set .difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
e = set(map(int, input().split()))
b = int(input())
f = set(map(int, input().split()))
print(len(e.difference(f)))

# Set .symmetric_difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
e = set(map(int, input().split()))
b = int(input())
f = set(map(int, input().split()))
print(len(e.symmetric_difference(f)))

# Set Mutations
# Enter your code here. Read input from STDIN. Print output to STDOUT
valoriA = int(input())
A = set(map(int, input().split()))
other = int(input())
for i in range(other):
    operation = input().split()
    rand = set(map(int, input().split()))
    if operation[0] == "update":
        A.update(rand)
    elif operation[0] == "intersection_update":
        A.intersection_update(rand)
    elif operation[0] == "difference_update":
        A.difference_update(rand)
    else:
        A.symmetric_difference_update(rand)
print(sum(A))

# The Captain's Room
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
size = int(input())
roomn = list(map(int, input().split()))
conto = Counter(roomn)
for i, repeats in conto.items():
    if repeats == 1:
        print(i)
        break

# Check Subset
# Enter your code here. Read input from STDIN. Print output to STDOUT
T = int(input())
for i in range(T):
    val_A = int(input())
    A = set(map(int, input().split()))
    val_B = int(input())
    B = set(map(int, input().split()))
    print(A <= B)

# Check Strict Superset
# Enter your code here. Read input from STDIN. Print output to STDOUT
A = set(map(int, input().split()))
n = int(input())
strict = True
for i in range(n):
    other = set(map(int, input().split()))
    if not A.issuperset(other):
        strict = False
        break
print(strict)

# collections.Counter()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
shoe = int(input())
size = Counter(list(map(int, input().split())))
custom=int(input())
money=0
for i in range(custom):
    shoe_size, x = map(int, input().split())
    if size[shoe_size]:
        money = money+x
        size[shoe_size]=size[shoe_size]-1
print(money)

# DefaultDict Tutorial
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict
A = defaultdict(list)
n, m = list(map(int, input().split()))
for i in range(n):
    A[input()].append(i+1)
for j in range(m):
    B = input()
    if B in A.keys():
        print(*A[B])
    else:
        print(-1)

# Collections.namedtuple()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple
stud = int(input())
col = namedtuple("col", input().split())
print(f"{sum(int(col(*input().split()).MARKS)for i in range(stud))/stud:.2f}")

# Collections.OrderedDict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
items = int(input())
dicto = OrderedDict()
for i in range(items):
    item, price = input().rsplit(" ", 1)
    dicto[item] = dicto.get(item, 0) + int(price)
for item, price in dicto.items():
    print(f"{item} {price}")

# Word Order
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
n = int(input())
word = [list(map(str, input().split()))for i in range(n)]   
x = [j[0]for j in word]
print(len(Counter(x).keys()))
freq = list(Counter(x).values())
str_freq = " ".join([str(y) for y in freq])
print(str_freq)

# Collections.deque()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
d = deque()
n = int(input())
for i in range(n):
    method = input().split()
    if method[0] == "append":
        d.append(method[1])
    elif method[0] == "pop":
        d.pop()
    elif method[0] == "popleft":
        d.popleft()
    else:
        d.appendleft(method[1])
print(" ".join(d))

# Piling Up!
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
T = int(input())
for i in range(T):
    n = int(input())
    d = deque(list(map(int, input().split())))
    x = 2**31
    for j in range(len(d)):
        if d[0]>=d[len(d)-1] and d[0]<=x:
            x = d.popleft()
        elif d[len(d)-1]<=x:
            x = d.pop()
        else:
            print("No")
            break
    if len(d) == 0:
        print("Yes")

# Company Logo
#!/bin/python3
from collections import Counter
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    s = input()
    sort = sorted(s)
    comm = Counter(list(sort))
    for i, x in comm.most_common(3):
        print(i, x)

# Calendar Module
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
m,d,y = map(int, input().split())
a = calendar.weekday(y,m,d)
giorno= list(calendar.day_name)
print(giorno[a].upper())

# Time Delta
#!/bin/python3
import math
import os
import random
import re
import sys
from datetime import datetime
#I find the difference in seconds within the timestamps
def time_delta(t1, t2):
    time_format = '%a %d %b %Y %H:%M:%S %z'
    dt1 = datetime.strptime(t1, time_format)
    dt2 = datetime.strptime(t2, time_format)
    # now the difference within the datetimes
    delta = abs((dt1 - dt2).total_seconds())
    return str(int(delta))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input()) 
    for t_itr in range(t):
        t1 = input().strip()
        t2 = input().strip()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n') 

# Exceptions
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    T = int(input()) 
    for _ in range(T):
        try:
            a, b = input().split()
            print(int(a) // int(b))
        except ZeroDivisionError as e:
            print("Error Code:", e)  
        except ValueError as e:
            print("Error Code:", e)

# Zipped!
# Enter your code here. Read input from STDIN. Print output to STDOUT
n,x = map(int, input().split())
grades = []
for i in range(x):
    grades.append(list(map(float, input().split())))
lista =list(zip(*grades))
for j in range(n):
    average= sum(list(lista[j]))/x
    print(average)

# Athlete Sort
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())
for i in sorted(arr, key = lambda x : x[k]):
    print(' '.join(str(y) for y in i))

# ginortS
# Enter your code here. Read input from STDIN. Print output to STDOUT
s = sorted(input())
upper=[]
lower=[]
odd=[]
even=[]
for i in s:
    if i.isalpha():
        if i.islower():
            a=lower
        else:
            a=upper
    else:
        if int(i)%2:
            a=odd
        else:
            a=even
    a.append(i)
print("".join(lower+upper+odd+even))

# HTML Parser - Part 1
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser
    
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        self.print_tag_attrs(attrs=attrs)
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        self.print_tag_attrs(attrs=attrs)
    def print_tag_attrs(self, attrs):
        [print(f"-> {i[0]} > {i[1]}") for i in attrs]
parser = MyHTMLParser()
parser.feed(''.join([input() for _ in range(int(input()))]))

# HTML Parser - Part 2
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if "\n" not in data:
            print(">>> Single-line Comment")
        else:
            print(">>> Multi-line Comment")
        print(data)
    def handle_data(self, data):
        if i := data.rstrip():
            print(">>> Data")
            print(i) 
  
  
  
  
  
  
  
  
  
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        if attrs:
            [print(f'-> {x[0]} > {x[1]}') for x in attrs]
    def handle_startendtag(self, tag, attrs):
        print(tag)
        if attrs:
            [print(f'-> {y[0]} > {y[1]}') for y in attrs]  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
  
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Validating UID
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
check = r"^(?=([A-Z]*[a-z\d]){2})(?=([a-zA-Z]*[\d]){3})(([a-zA-Z0-9])(?!.*\4)){10}$"
t = int(input())
for i in range(t):
    UID = input()
    x = re.match(check, UID)
    if x:
        print("Valid")
    else:
        print("Invalid")

# Validating Credit Card Numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
contr = r"(?=[456]\d{3}(-?\d{4}){3}$)(?!.*(.)-?(\2-?){3})"
for i in range(n):
    cr = input()
    x = re.match(contr, cr)
    if x:
        print("Valid")
    else:
        print("Invalid")

# Validating Postal Codes
regex_integer_in_range = r"^([1-9][0-9]{5})$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(?=(.)(.)(\1))"	# Do not delete 'r'.

# Detect Floating Point Number
# Enter your code here. Read input from STDIN. Print output to STDOUT
test = int(input())
for i in range(test):
    s = input()
    f = True
    try:
        float_s = float(s)
        if s.count(".")==0:
            f = False
    except:
        f = False   
    print(f)

# Re.split()
regex_pattern = r"[.,]"	# Do not delete 'r'.

# Group(), Groups() & Groupdict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
S = input()
res = re.search(r'([a-zA-Z0-9])\1', S)
if res:
    print(res.group(1))
else:
    print(-1)

# Re.findall() & Re.finditer()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
S = input()
res = re.findall(r'(?<=[^aeiouAEIOU])[aeiouAEIOU]{2,}(?=[^aeiouAEIOU])',S)
if res:
    for i in res:
        print(i)
else:
    print(-1)

# Re.start() & Re.end()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
S = input()
k = input()
def results(S, k):
    f = re.finditer(fr'(?=({k}))', S)
    x = [(match.start(1), match.end(1)-1) for match in f]
    return x if x else [(-1, -1)]
if results(S, k):
    for i in results(S, k):
        print(i)
else:
    print((-1, -1)) 

# Regex Substitution
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
for i in range(n):
  s = input()
  s = re.sub(r"(?<= )&&(?= )", "and", s)
  s = re.sub(r"(?<= )\|\|(?= )", "or", s)
  print(s)

# Validating Roman Numerals
regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"	# Do not delete 'r'.

# Validating phone numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
for i in range(n):
    S = input().strip()
    if re.match(r'^[789]\d{9}$', S):
        print("YES")
    else:
        print("NO")

# Validating and Parsing Email Addresses
# Enter your code here. Read input from STDIN. Print output to STDOUT
import email.utils
import re
n= int(input())
for i in range(n):
    auaua = email.utils.parseaddr(input())
    contr = re.match(r'^[a-zA-Z][\w._-]+@[a-zA-Z]+\.[a-zA-Z]{1,3}$', auaua[1])
    if contr:
        print(email.utils.formataddr(auaua))

# Hex Color Code
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
for i in range(n):
    CSS = input().strip()
    a = re.findall(r"(#[0-9a-fA-F]{3}|#[0-9a-fA-F]{6})(?=[;),])", CSS)
    if a :
        print(*a, sep="\n")

# Matrix Script
#!/bin/python3
import math
import os
import random
import re
import sys
first = input().rstrip().split()
n = int(first[0])
m = int(first[1])
matrix = []
for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
decoded_string = ""
for col in range(m):
    for row in range(n):
        decoded_string += matrix[row][col]
final_string = re.sub(r'(?<=\w)([^\w]+)(?=\w)', ' ', decoded_string)
print(final_string)

# XML 1 - Find the Score

def get_attr_number(node):
    # your code goes here
    attr_count = 0
    for elem in node.iter():
        attr_count += len(elem.attrib)
    return attr_count

# XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    # your code goes here
    if level + 1 > maxdepth:
        maxdepth = level + 1
    for child in elem:
        depth(child, level + 1)

# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        # complete the function
        form_numbers = []
        for number in l:
            form_number = number[-10:] 
            form_numbers.append(f"+91 {form_number[:5]} {form_number[5:]}")
        return f(form_numbers)
    return fun

# Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        # complete the function
        return [f(person) for person in sorted(people, key=operator.itemgetter(2))]
    return inner

# Arrays

def arrays(arr):
    # complete this function
    # use numpy.array
    return numpy.array(arr, float)[::-1]

# Shape and Reshape
import numpy
arr = input().split(' ')
arr1=numpy.array(arr,int)
print(numpy.reshape(arr1,(3,3)))


# Transpose and Flatten
import numpy
N, M = map(int, input().split())
array = numpy.array([input().split() for i in range(N)], int)
print(numpy.transpose(array))
print(array.flatten())    



# Concatenate
import numpy
N,M,P=input().split()
N=int(N)
M=int(M)
P=int(P)
arr1= numpy.array([input().split() for i in range(N)],int)
arr2= numpy.array([input().split() for j in range(M)],int)
print(numpy.concatenate((arr1,arr2),axis=0))


# Zeros and Ones
import numpy
dim = map(int, input().split())
dims=tuple(dim)
print(numpy.zeros(dims, dtype= int))
print(numpy.ones(dims, dtype=int))


# Eye and Identity
import numpy
numpy.set_printoptions(legacy='1.13')
N,M=input().split()
print(numpy.eye(int(N),int(M),k=0))


# Array Mathematics
import numpy
N,M=input().split()
N=int(N)
M=int(M)
A=numpy.array([input().split() for i in range(N)],int)
B=numpy.array([input().split() for i in range(N)],int)
print(A+B)
print(A-B)
print(A*B)
print(A//B)
print(A%B)
print(A**B)

# Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')
A=numpy.array(input().split(),float)
print(numpy.floor(A))
print(numpy.ceil(A))
print(numpy.rint(A))

# Sum and Prod
import numpy
N,M=input().split()
N=int(N)
M=int(M)
arr=numpy.array([input().split()for i in range(N)],int)
arr_added=numpy.sum(arr,axis=0)
arr_prod=numpy.prod(arr_added)
print(arr_prod)


# Min and Max
import numpy
N,M=input().split()
N=int(N)
M=int(M)
arr=numpy.array([input().split()for i in range(N)],int)
minimus=numpy.min(arr,axis=1)
print(numpy.max(minimus))

# Mean, Var, and Std
import numpy
N,M=input().split()
N=int(N)
M=int(M)
arr=numpy.array([input().split()for i in range(N)],int)
print(numpy.mean(arr,axis=1))
print(numpy.var(arr,axis=0))
print(numpy.std(arr,axis=None))


# Dot and Cross
import numpy
N=input()
N=int(N)
A=numpy.array([input().split()for i in range(N)],int)
B=numpy.array([input().split()for i in range(N)],int)

print(numpy.dot(A,B))
#print(numpy.cross(A,B))


# Inner and Outer
import numpy
A=numpy.array([input().split()for i in range(1)],int)
B=numpy.array([input().split()for i in range(1)],int)
print(numpy.inner(A,B)[0,0])
print(numpy.outer(A,B))

# Polynomials
import numpy
P=input().split()
x=input()
P=numpy.array(P,float)
x=float(x)
print(numpy.polyval(P, x))

# Linear Algebra
import numpy
N=input()
N=int(N)
A=numpy.array([input().split()for i in range(N)],float)
print(round(numpy.linalg.det(A),2))  

# Tuples
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
t = tuple(map(int, input().split()))
print(hash(t))

# Birthday Cake Candles
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#
def birthdayCakeCandles(candles):
    # Write your code here
    l=[]
    massimo=max(candles)
    for j in candles:
        if j==massimo:
            l.append(j)
    return len(l)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

# Number Line Jumps
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#
def kangaroo(x1, v1, x2, v2):
    # Write your code here
    for n in range(5000):
            if x1 +n*v1 == x2+n*v2:
                return "YES"
    else:
        return "NO"
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()

# Viral Advertising
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#
def viralAdvertising(n):
    # Write your code here
    peo=5
    cum=0
    for i in range(n):
        like=peo//2
        cum +=like
        peo=like*3
    return cum
        
    return n//2
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()

# Recursive Digit Sum
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#
def superDigit(n, k):
    # Write your code here
    summa1=0
    for i in n:
            summa1+=int(i)
    summap = summa1 * k
    while summap >= 10:
        summap = sum(int(i) for i in str(summap))
    return summap
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()

# Insertion Sort - Part 1
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort1(n, arr):
    # Write your code here
    k=arr[n-1]
    count=0
    for i in range(1,n):
        if arr[n-i-1]>k:
            arr[n-i]=arr[n-i-1]
            count+=1
            print(" ".join(map(str, arr)))
    arr[n-count-1]=k
    print(" ".join(map(str, arr)))
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

# Insertion Sort - Part 2
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort2(n, arr):
    # Write your code here
    for i in range(1, n):
        k = arr[i]  
        j = i - 1
        while j >= 0 and arr[j] > k:
            arr[j + 1] = arr[j] 
            j -= 1
        arr[j + 1] = k
        print(" ".join(map(str, arr)))
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)

