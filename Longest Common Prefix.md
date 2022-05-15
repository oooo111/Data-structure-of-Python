``` python
# Question

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".


Link：https://leetcode.cn/problems/longest-common-prefix


#Example:
	
Input: strs = ["flower","flow","flight"]
Output: "fl"

   
Link：https://leetcode.cn/problems/Roman to Integer
    
## method
def longestCommonPrefix(self, strs):
    str1=""
    for temp in zip(*strs):
        temps=len(set(temp))
        if temps==1:
            str1+=temp[0]
        else:
            break
    return str1
```



