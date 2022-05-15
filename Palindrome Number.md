``` python
# Question
Given an integer x, return true if x is palindrome integer.An integer is a palindrome when it reads the same backward as forward.For example, 121 is a palindrome while 123 is not.
 
    
#Example:
Input: x = 121
Output: true
Explanation: 121 reads as 121 from left to right and from right to left.

   
Link：https://leetcode.cn/problems/palindrome-number
    
## method
def isPalindrome(x):
    li=[]
    for i in str(x):
        li.append(i)
    if li==li[::-1]:
        return True
    else:
        return False
```



