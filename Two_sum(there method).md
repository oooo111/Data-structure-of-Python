``` python
#Question
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

#Example
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Link：https://leetcode.cn/problems/two-sum


## method 1 hashtable

def twosum1(nums, target):
    dic={}
    for i,j in enumerate(nums):
        if dic.get(target-j) is not None:
            return [i,dic.get(target-j)]
        dic[j]=i
        
        
## method 2 exhaustive search
def twosum2(nums,target):
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
            if nums[i]+nums[j]==target:
                return [i,j]

           
## method 3 binary search

def two_sum3(nums, target):
    for i, number in enumerate(numbers):   
        second_val = target - number
        low, high =i+1, len(numbers)-1
        while low <= high:
            mid = (low+high)//2
            if second_val == numbers[mid]:
                return [i, mid]

            if second_val > numbers[mid]:
                low = mid + 1
                else:
                    high = mid - 1
    return None


            
   

```



