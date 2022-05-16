``` python
#172. Factorial Trailing Zeroes

# Question
Given an integer n, return the number of trailing zeroes in n!.
Note that n! = n * (n - 1) * (n - 2) * ... * 3 * 2 * 1.

# Example
Input: n = 3
Output: 0
Explanation: 3! = 6, no trailing zero.

Link：https://leetcode.cn/problems/factorial-trailing-zeroes
    
# code    step1:求出这个数的阶乘等于多少 step2:第二步换成字符串把每个元素添加到列表 step3:第三步判断尾部0的数量


def trailingZeroes(n):
    total_sum=1
    count=0
    li=[]
    if n==0:
        return 0
    for i in range(1,n+1):
        total_sum*=i
        total_sum=str(total_sum)
    for i in total_sum:
        li.append(i)
    for i in li[::-1]:
        if int(i)==0:
            count+=1
        else:
            break
    return count


        

```

