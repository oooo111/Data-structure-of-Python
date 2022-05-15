``` python
# Question
Given a string s, find the length of the longest substring without repeating characters.

#Example :
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Link：https://leetcode.cn/problems/longest-substring-without-repeating-characters
    
## methode 1 double pointer

def lengthOfLongestSubstring(self, s):
    ans=0
    hashset=set()
    for i in range(len(s)):
        count=0
        for j in range(i,len(s)):
            if s[j] in hashset:
                hashset.clear()
                break
            else:
                hashset.add(s[j])
                count+=1
        ans=max(ans,count)
    return ans

### method2 sliding window

def lengthOfLongestSubstring(self, s):
    occ=set()
    n=len(s)
    rk,ans=-1,0
    for i in range(n):
        if i!=0:
            occ.remove(s[i-1])
        while rk+1<n and s[rk+1] not in occ:
            occ.add(s[rk+1])
            rk+=1
        ans=max(ans,rk-i+1)
    return ans
```



