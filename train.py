from random import randint
llist=["有效的括号","最小栈","字符串解码","739. 每日温度","84 柱状图中最大的矩形","85 最大矩形","42. 接雨水","树的非递归遍历后序双栈法"]
print(llist[randint(0,len(llist)-1)])


#全排列
# def compute(llist):
# print(compute([1,2,3]))

#单词搜索
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = "ABCCEC"
# def solution(board,word):
# print(solution(board,word))

#分割字符串
# def partition(s)
# print(partition("aaba"))

#分割等和子集
# def fun(strs):
#print(fun([1, 5, 11, 5]))



#最长公共子序列
# def longestCommonSubsequence(text1, text2):
# print(longestCommonSubsequence("abcde","ace"))

#01背包
# def onezerobag(N,W,wt,val):
# print(onezerobag(3,4,[2,1,3],[4,2,3]))
#N和W分别是背包能装物体的数量和背包能装的重量。wt数组指的是物体的重量，val指的是对应的价值。
# class Solution:
#     def isValid(self, s: str) -> bool:
#         dic={'{':'}','(':')','[':']','?':'?'}
#         ss=['?']
#         for c in s:
#             if c in dic:
#                 ss.append(c)
#             else:
#                if dic[ss.pop()]!=c:return False 
#         return len(ss)==1
# ss=Solution()
# print(ss.isValid('?#'))
class Solution:
    def isValid(self, s: str) -> bool:
        dic={'{':'}','[':']','(':')','?':'?'}
        stack=['?']
        for i in s:
            if i in dic:stack.append(i)
            elif dic[stack.pop()]!=i :return False
        return len(stack)==1
ss=Solution()
print(ss.isValid("{{[}]}"))




            
            
