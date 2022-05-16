``` python
# 187. Repeated DNA Sequences

# Question 
The DNA sequence is composed of a series of nucleotides abbreviated as 'A', 'C', 'G', and 'T'.
For example, "ACGAATTCCG" is a DNA sequence.
When studying DNA, it is useful to identify repeated sequences within the DNA.
Given a string s that represents a DNA sequence, return all the 10-letter-long sequences (substrings) that occur more than once in a DNA molecule. You may return the answer in any order.

# Example
Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
Output: ["AAAAACCCCC","CCCCCAAAAA"]

Link：https://leetcode.cn/problems/repeated-dna-sequences
    
## Code Hashtable


def findRepeatedDnaSequences(s):
    li=[]
    L=10 # 设定需要寻找字符串的长度
    hashset=defaultdict(int)  #先将字典定义成整型
    for i in range(len(s)-L+1):  #遍历 len(s)-L+1次
        sub=s[i:i+L]     # 记录子串
        hashset[sub]+=1
        if hashset[sub]==2:
            li.append(sub)
    return li


### improve version but need more complex time


def findRepeatedDnaSequences(s):
    li=[]
    L=10
    hashset=defaultdict(int)
    for i in range(len(s)-L+1):
        sub=s[i:i+L]
        hashset[sub]+=1
        for key,value in hashset.items():
            if value>1:
                li.append(key)
    return list(set(li))







```

