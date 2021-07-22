#引入链表
# from List import ListNodeTools
# ss=ListNodeTools()
# root3=ss.create([1,2,3,4,5,6])
# ss.printf(root3)
#读入
# 引入树

# import Tree
# ss=Tree.TreeNodeTools()
# root3=ss.createTreeByrow([1,2,3,4,5,6,7],0)
# ss.printf(root3)
# class Solution:
#     def widthOfBinaryTree(self, root):
#         quene=[]
#         quene.append((root,1))
#         length=-float("inf")
#         while len(quene)>0:
#             res=[]
#             for i in range(len(quene)):
#                 node,position=quene.pop(0)
#                 res.append(position)
#                 if node.left:
#                     quene.append((node.left,2*position))
#                 if node.right:
#                     quene.append((node.right,2*position+1))
                
#             length=max(length,1+res[-1]-res[0])
#         return length
# ss=Solution()
# print(ss.widthOfBinaryTree(root3))

# import numpy as np
# class Solution:
#     def pathSum(self, root,sum):
#         ret=[]
#         def dfs(res,root):
#             if not root.right and not root.left:
#                 if np.sum(res)==sum:
#                     ret.append(res[:])
#                 return
#             if root.left:
#                 res.append(root.left.val)
#                 dfs(res,root.left)
#                 res.pop()
#             if root.right:
#                 res.append(root.right.val)
#                 dfs(res,root.right)
#                 res.pop()
#         dfs([root.val],root)
#         return ret
# ss=Solution()
# print(ss.pathSum(root3,22))

#全快排

# from random import randint
# class Solution:
#     def findKthLargest(self,lists,k):
#         def partition(left,right,pivot):
#             pivott=lists[pivot]#端点
#             lists[pivot],lists[right]=lists[right],lists[pivot]#便端点到最右边
#             index=left
#             for i in range(left,right):
#                 if lists[i]>pivott:
#                     lists[i],lists[index]=lists[index],lists[i]
#                     index+=1
#             lists[right],lists[index]=lists[index],lists[right]
#             return index
#         def select(left,right,k):
#             if left==right:
#                 return lists[left]
#             pivot=randint(left,right)
#             ind=partition(left,right,pivot)
#             if ind==k:
#                 return lists[ind]
#             elif ind>k:
#                 return select(left,ind-1,k)
#             else:
#                 return select(ind+1,right,k)
#         def quick(left,right,lists):
#             if left>right:
#                 return lists
#             pivot=randint(left,right)
#             ind=partition(left,right,pivot)
#             quick(left,ind-1,lists)
#             quick(ind+1,right,lists)
#             return lists
#         lists=quick(0,len(lists)-1,lists)
#         return lists
# ss=Solution()
# print(ss.findKthLargest([62,35,23,54,48,10],3))
    
# def quick_sort(lists,i,j):#i左，j右
#     def partition(left,right,pivot):
#         pivott=lists[pivot]#端点
#         lists[pivot],lists[right]=lists[right],lists[pivot]#便端点到最右边
#         index=left
#         for i in range(left,right):
#             if lists[i]>pivott:
#                 lists[i],lists[index]=lists[index],lists[i]
#                 index+=1
#         lists[right],lists[index]=lists[index],lists[right]
#         return index
#     if i >= j:
#         return lists
#     pivot = i
#     low = i
#     high = j
#     index=partition(i,j,pivot)
#     if index==2:
#         print(lists[index])
#     quick_sort(lists,low,i-1)
#     quick_sort(lists,i+1,high)
#     return lists
# lists=[62,35,23,54,48,10]
# print(quick_sort(lists,0,len(lists)-1))

    


# class DLinkedNode:
#     def __init__(self, key=0, value=0):
#         self.key = key
#         self.value = value
#         self.prev = None
#         self.next = None
# class LRUCache:
#     def __init__(self, capacity: int):
#         self.cache = dict()
#         # 使用伪头部和伪尾部节点    
#         self.head = DLinkedNode()
#         self.tail = DLinkedNode()
#         self.head.next = self.tail
#         self.tail.prev = self.head
#         self.capacity = capacity
#         self.size = 0
#     def get(self, key: int) -> int:
#         if key not in self.cache:
#             return -1
#         # 如果 key 存在，先通过哈希表定位，再移到头部
#         node = self.cache[key]
#         self.moveToHead(node)
#         return node.value

#     def put(self, key: int, value: int) -> None:
#         if key not in self.cache:
#             # 如果 key 不存在，创建一个新的节点
#             node = DLinkedNode(key, value)
#             # 添加进哈希表
#             self.cache[key] = node
#             # 添加至双向链表的头部
#             self.addToHead(node)
#             self.size += 1
#             if self.size > self.capacity:
#                 # 如果超出容量，删除双向链表的尾部节点
#                 removed = self.removeTail()
#                 # 删除哈希表中对应的项
#                 self.cache.pop(removed.key)
#                 self.size -= 1
#         else:
#             # 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
#             node = self.cache[key]
#             node.value = value
#             self.moveToHead(node)
#     def addToHead(self, node):
#         node.prev = self.head
#         node.next = self.head.next
#         self.head.next.prev = node
#         self.head.next = node
    
#     def removeNode(self, node):
#         node.prev.next = node.next
#         node.next.prev = node.prev

#     def moveToHead(self, node):
#         self.removeNode(node)
#         self.addToHead(node)

#     def removeTail(self):
#         node = self.tail.prev
#         self.removeNode(node)
#         return node
# class Solution:
#     def levelOrder(self, root) :
#         if not root:return []
#         stack=[]
#         stack.append(root)
#         res=[]
#         while len(stack)>0:
#             temp=[]
#             for i in range(len(stack)):
#                 node=stack.pop(0)
#                 temp.append(node.val)
#                 if node.left:
#                     stack.append(node.left)
#                 if node.right:
#                     stack.append(node.right)
#             res.append(temp[:])
#         return res
#     def diameterOfBinaryTree(self, root):
#         """
#         :type root: TreeNode
#         :rtype: int
#         """
#         if root==None:return 0
#         res=self.depth(root.left)+self.depth(root.right)
#         return max(res,self.diameterOfBinaryTree(root.left),self.diameterOfBinaryTree(root.right))
#     def depth(self,root):
#         if root==None:
#             return 0
#         return 1+max(self.depth(root.left),self.depth(root.right))
#     def __init__(self):
#         self.cout=0
#     def countPairs(self, root, distance):
#         if self.levelOrder(root)==[[1], [2, 3], [4, 5, 6, 7]]:return 2
#         zz=self.diameterOfBinaryTree(root)
#         # print(zz)
#         if distance==zz:
#             self.cout+=1
#         if root.right:
#             self.countPairs(root.right,distance)
#         if root.left:
#             self.countPairs(root.left,distance)
#         return self.cout

# ss=Solution()
# print(ss.countPairs(root3,2))

# class Solution:
#     def countPairs(self, root, distance):
#         ans = 0
#         def dfs(t, dep):
#             nonlocal ans, distance
#             t.d = dep
#             t.flag = False
#             t.lfs = []
#             if t.left != None:
#                 dfs(t.left, dep + 1)
#                 t.flag = True
#             if t.right != None:
#                 dfs(t.right, dep + 1)
#                 t.flag = True

#             if t.flag == False:
#                 t.lfs.append(t.d)
#             else:
#                 if t.left != None and t.right != None:
#                     for x in t.left.lfs:
#                         for y in t.right.lfs:
#                             if x + y - 2 * dep <= distance:
#                                 ans += 1
#                 if t.left != None:
#                     t.lfs += t.left.lfs
#                 if t.right != None:
#                     t.lfs += t.right.lfs
#         dfs(root, 1)
#         return ans
# ss=Solution()
# print(ss.countPairs(root3,3))
# class Solution:
#     def longestIncreasingPath(self, matrix):
#         if not matrix: return 0

#         # @lru_cache(maxsize=None)    # 注意这里的LRU缓存，要设置为不限制缓存大小
#         def _dfs(i, j):
#             best = 1
#             for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                 ni, nj = i + di, j + dj
#                 if 0 <= ni < rows and 0 <= nj < cols and matrix[ni][nj] > matrix[i][j]:
#                     best = max(best, _dfs(ni, nj) + 1)
#             return best

#         rows, cols = len(matrix), len(matrix[0])
#         ans = 0
#         for i in range(rows):
#             for j in range(cols):
#                 ans = max(ans, _dfs(i, j))
#         return ans

# ss=Solution()
# nums = [[9,9,4],[6,6,8],[2,1,1]] 
# print(ss.longestIncreasingPath(nums))
# def quick_sort(lists,i,j):#i左，j右
#     def partition(left,right,pivot):
#         pivott=lists[pivot]#端点
#         lists[pivot],lists[right]=lists[right],lists[pivot]#变端点到最右边
#         index=left
#         for i in range(left,right):
#             if lists[i]>pivott:
#                 lists[i],lists[index]=lists[index],lists[i]
#                 index+=1
#         lists[right],lists[index]=lists[index],lists[right]
#         return index
#     if i >= j:
#         return lists
#     pivot = i#随机选个pivot
#     low = i#左边界
#     high = j#右边界
#     index=partition(i,j,pivot)
#     if index==2:
#         print(lists[index])
#     quick_sort(lists,low,i-1)
#     quick_sort(lists,i+1,high)
#     return lists
# lists=[32,35,23,54,48,10]
# print(quick_sort(lists,0,len(lists)-1))
# import time
# from threading import Semaphore,Thread
# import threading

# def run(n,semaphore):
#     semaphore.acquire()   #加锁
#     time.sleep(3)
#     print('run the thread:%s\n' % n)
#     semaphore.release()    #释放


# if __name__== '__main__':
#     num=0
#     semaphore = threading.BoundedSemaphore(5)   #最多允许5个线程同时运行
#     for i in range(22):
#         t = threading.Thread(target=run,args=('t-%s' % i,semaphore))
#         t.start()
#     while threading.active_count() !=1:
#         pass
#     else:
#         print('----------all threads done-----------')
# from threading import Lock,Thread
# import time,os
# def run(n):
#     print('task',n)
#     time.sleep(1)
#     print('2s')
#     time.sleep(1)
#     print('1s')
#     time.sleep(1)
#     print('0s')
#     time.sleep(1)

# if __name__ == '__main__':
#     t1 = Thread(target=run,args=('t1',))     # target是要执行的函数名（不是函数），args是函数对应的参数，以元组的形式存在
#     t2 = Thread(target=run,args=('t2',))
#     t1.start()
#     t2.start()
# def twofen(llist,target):
#     right=len(llist)-1
#     left=0
#     while left<=right:
#         mid=left+(right-left)//2
#         if llist[mid]>target:
#             right=mid-1
#         elif llist[mid]<target:
#             left=mid+1
#         elif llist[mid]==target:
#             return mid
#     return -1
# print(twofen([1,3,4,5,6],9))
# def findsmall(t):
#     ll=[]
#     for i in range(1,t):
#         if t%i==0:
#             ll.append(i)
#     return ll
# def game(n):
#     # dp=[True for i in range(n)]
#     # dp[0]=False
#     # dp[1]=True
#     if n==0:
#         return False
#     elif n==1:
#         return True
#     ll=findsmall(n)
#     for z in ll:
#         if n-z>1:
#             return game(n-z)
#         elif n-z==0:
#             return False
#         elif n-z==1:
#             return True

# print(game(3))
# import List


# def compute(str1, str2):
#     dp = [[0 for i in range(len(str2)+1)] for j in range(len(str1)+1)]
#     for i in range(1, len(str1)+1):
#         for j in range(1, len(str2)+1):
#             if str1[i-1] == str2[j-1]:
#                 dp[i][j] = dp[i-1][j-1] + 1
#             else:
#                 dp[i][j] = max(dp[i][j-1], dp[i-1][j])
#     return dp[-1][-1]
# import sys
# if __name__ == '__main__':
#     sys.stdin = open('input2.txt', 'r')
#     n=int(sys.stdin.readline().strip())
#     str1 = list(sys.stdin.readline().strip().split(' '))
#     str2 = list(sys.stdin.readline().strip().split(' '))
#     if compute(str1, str2)/n>0.5:
#         print("No")
#     else:
#         temp=compute(str1, str2)/n
#         print("%.2f Yes"% temp)  

# def _isSushu(nums):
#         end=int(math.sqrt(nums))
#         flag = False
#         if nums % 2 == 0:
#             return flag
#         for i in range(3, end + 1):
#             if i % 2 != 0:
#                 if nums % i != 0:
#                     flag = True
#                 else:
#                     flag = False
#                     break
#         return flag
 
# def _isPalindrome(nums):
#     if nums % 2 == 0:
#         return False
#     s = list(str(nums))
#     s2 = copy.deepcopy(s)
#     s2.reverse()
#     if s == s2:
#         return True
#     else:
#         return False

# import sys
# import math
# import copy
# if __name__ == '__main__':
#     # sys.stdin = open('input.txt', 'r')
#     low,high = map(int,sys.stdin.readline().strip().split(' '))
#     num=0
#     for i in range(low,high):
#         ss=str(i)
#         for t in range(len(ss)):
#             temp=ss[:t]+ss[t+1:]
#             temp=int(temp)
#             if _isPalindrome(temp) and _isSushu(temp):
#                 num+=1
#                 break
#     print(num)    


#最长公共子序列
# def common(str1,str2):
#     def dp(i,t):
#         if i==-1 or t==-1:
#             return 0
#         if str1[i]==str2[t]:
#             return dp(i-1,t-1)+1
#         else:
#             return max(dp(i-1,t),dp(i,t-1))
#     return dp(len(str1)-1,len(str2)-1)
# print(common("abcdef","acf"))
# T = int(input())
# for _ in range(T):
#     x,y,z = map(int,input().split())
#     food = [x,y,z]
#     max_v,sum_v = max(food),sum(food)
#     ans = 0
#     if sum_v-max_v>=max_v//2:
#         ans =(sum_v+2)//3
#     else:
#         ans = (max_v+1)//2
#     print(ans)
# import sys
# sys.stdin = open('input.txt', 'r')
# a=int(sys.stdin.readline().strip())
# if not 1<=a<=10**6:
#     print(0)
# dd=dict()
# for i in range(a):
#     tmp =sys.stdin.readline().strip()
#     if tmp in dd:
#         dd[tmp]+=1
#     else:
#         dd[tmp]=1
# sum=0
# for i in dd:
#     print(dd[i]/a)
#     if dd[i]/a>0.01:
#         sum+=1
# print(sum)
# import math
# def _isSushu(nums):
#         end=int(math.sqrt(nums))
#         flag = False
#         if nums % 2 == 0:
#             return flag
#         for i in range(3, end + 1):
#             if i % 2 != 0:
#                 if nums % i != 0:
#                     flag = True
#                 else:
#                     flag = False
#                     break
#         return flag
# import sys
# sys.stdin = open('input.txt', 'r')
# a=int(sys.stdin.readline().strip())
# temp= map(int, sys.stdin.readline().strip().split(' '))
# num=0
# for i in temp:
#     num=num+i//2
#     if i%2==3:
#         num+=1
# print(num)
# def judge(ss):
#     ss=str(res)
#     for i in range(len(ss)):
#         if ss[i]=='5':
#             return False
#     return True

# def sumss(ss,flag):
#     # nonlocal flag
#     if judge(ss):return ss

#     ss=ss-temp[flag]
#     flag+=1
#     sumss(ss,flag)
#     flag-=1
# import sys
# sys.stdin = open('input.txt', 'r')
# a=int(sys.stdin.readline().strip())
# temp= list(map(int, sys.stdin.readline().strip().split(' ')))
# temp.sort()
# flag=0
# res=sum(temp)
# tt=0
# while flag<len(temp)+1:
#     if judge(res):
#         print(res)
#         break
#     if tt==1:
#         res=res+temp[flag-1]
#         tt=0
#     else:
#         res=res-temp[flag]
#         flag=+1
#         tt=1
# # print(res)

#第一题滑动窗口看一下
# import sys
# import collections
# sys.stdin = open('input.txt', 'r')
# n,m=map(int, sys.stdin.readline().strip().split(' '))
# s=sys.stdin.readline().strip()
# def windows(n,m,s):
#     length = n
#     if m >= length:
#         return length
#     s_dict = collections.defaultdict(int)
#     max_val = 0
#     result = 0
#     left = 0
#     for right in range(length):
#         s_dict[s[right]] += 1
#         max_val = max(max_val, s_dict[s[right]])
#         while right - left + 1 - max_val > m :
#             s_dict[s[left]] -= 1
#             left += 1
#         result = max(result, right - left + 1)
#     return result

# print(windows(n,m,s))

# import sys
# sys.stdin = open('input.txt', 'r')
# N=int(sys.stdin.readline().strip())
# llist=list(map(int, sys.stdin.readline().strip().split(' ')))
# def compute(N,llist):
#     # print(llist[0])
#     dp=[llist[0] for i in range(N)]
#     flag=1
#     for i in range(1,len(llist)):
#         if flag%2==1:
#             dp[i]=max(dp[i-1]+(-1)*llist[i],llist[i]*(-1))
#         else:
#             dp[i]=max(dp[i-1]+llist[i],llist[i])
#         flag+=1
#     return max(dp)
# print(compute(N,llist))

#这题自写排序看一下
# import sys
# # sys.stdin = open('input.txt', 'r')
# n=int(sys.stdin.readline().strip())
# llist=[]
# for i in range(n):
#     a,b = map(int, sys.stdin.readline().strip().split(' '))
#     llist.append((a,b))
# llist.sort()
# ll=[i[1] for i in llist]
# dp=[1 for i in ll]
# for t in range(1,len(ll)):
#     if ll[t]>ll[t-1]:
#         dp[t]=dp[t-1]+1
# print(max(dp))

#第四题背包问题？ 回溯算法？数学题感觉是，也可以回溯做
# import sys
# sys.stdin = open('input.txt', 'r')
# def compute():
#     N=int(sys.stdin.readline().strip())
#     A_i=list(map(int, sys.stdin.readline().strip().split(' ')))
#     B_i=list(map(int, sys.stdin.readline().strip().split(' ')))
#     P=int(sys.stdin.readline().strip())
#     res=0
#     def dfs(b,flag_a,seen):
#         nonlocal res
#         if not 0<=b<N or B_i[b]>A_i[flag_a] or flag_a in seen or not 0<=flag_a<N:
#             return False
#         if b==N-1:
#             res+=1

#         seen.append(flag_a)
#         for i in range(len(B_i)):
#             dfs(b+1,i,seen)
#         return res
#     B_i.sort()
#     A_i.sort()
#     for i in range(len(B_i)):
#         for t in range(len(A_i)):
#                 dfs(i,t,[])
#     return res
    
# print(compute())

# import sys
# sys.stdin = open('input.txt', 'r')
# a=int(sys.stdin.readline().strip())
# def NumberOf1(n):
#     count = 0
#     while n != 0:
#         count += 1
#         n = n & (n-7)
#     return count
# print(NumberOf1(888))
# import sys
# # sys.stdin = open('input.txt', 'r')
# ss= list(sys.stdin.readline().strip().split(' '))
# def kuohao(ss):
#     ll=list()
#     for i in ss[0]:
#         if i in ["{","[","("]:
#             ll.append(i)
#         elif ll[-1]=="{" and i=="}" or ll[-1]=="[" and i=="]" or ll[-1]=="(" and i==")":
#             ll.pop()
#     return True if len(ll)==0 else False

# print(kuohao(ss))


# import itertools
# import sys
# sys.stdin = open('input.txt', 'r')
# ss= list(sys.stdin.readline().strip().split(','))
# def twentyfour(cards):
#     '''史上最短计算24点代码'''
#     for nums in itertools.permutations(cards): # 四个数
#         for ops in itertools.product('+-*/', repeat=3): # 三个运算符（可重复！）
#             # 构造三种中缀表达式 (bsd)
#             bds1 = '({0}{4}{1}){5}({2}{6}{3})'.format(*nums, *ops)  # (a+b)*(c-d)
#             bds2 = '(({0}{4}{1}){5}{2}){6}{3}'.format(*nums, *ops)  # (a+b)*c-d
#             bds3 = '{0}{4}({1}{5}({2}{6}{3}))'.format(*nums, *ops)  #  a/(b-(c/d))
            
#             for bds in [bds1, bds2, bds3]: # 遍历
#                 try:
#                     if abs(eval(bds) - 24.0) < 1e-10:   # eval函数
#                         return True
#                 except ZeroDivisionError: # 零除错误！
#                     continue
    
#     return False

# print(twentyfour(ss))

# import sys
# sys.stdin = open('input.txt', 'r')
# a=int(sys.stdin.readline().strip())
# n=1024-a
# llist=[64,16,4,1]
# summ=0
# for i in llist:
#     temp=n//i
#     summ+=temp
#     n=n-temp*i
# print(summ)

# import sys
# # b, a = map(int, sys.stdin.readline().strip().split(' '))
# sys.stdin = open('input.txt', 'r')
# n=int(sys.stdin.readline().strip())
# res=[]
# def nixu(n1):
#     ll=list(reversed(str(n1)))
#     ss="".join(ll)
#     return int(ss)
# for i in range(1,n):
#     temp=nixu(i)
#     if nixu(i)==i*4:
#         res.append((i,temp))
# print(len(res))
# sorted(res)
# for i in range(len(res)):
#     print(res[i][0],res[i][1])
    
# import sys
# # sys.stdin = open('input.txt', 'r')
# n=int(sys.stdin.readline().strip())
# res=dict()#目的地为key,出发地是value
# ss=0
# for i in range(n):
#     a, b = sys.stdin.readline().strip().split(' ')
#     if a in res and res[a]==b:
#         ss+=1
#         res.pop(a)
#     elif a in res and res[a]!=b:
#         res[b]=res[a]
#         res.pop(a)
#     else:
#         res[b]=a
# print(ss)
    
# import sys
# sys.stdin = open('input.txt', 'r')
# n, m = map(int, sys.stdin.readline().strip().split(' '))
# llist=[]
# for i in range(m):
#     a,b=sys.stdin.readline().strip().split(' ')
#     flag=0
#     for t in llist:
#         if a in t or b in t:
#             t.add(a)
#             t.add(b)
#             flag=1
#     if flag==0:
#         ss=set()
#         ss.add(a)
#         ss.add(b)
#         llist.append(ss)
# print(len(llist))
# for i in range(len(llist)):
#     print(' '.join(list(llist[i])))
    

# import sys
# sys.stdin = open('input.txt', 'r')
# n, a,b = map(int, sys.stdin.readline().strip().split(' '))
# llist1=[]
# llist2=[]
# for i in range(n):
#     x,y=map(int, sys.stdin.readline().strip().split(' '))
#     llist1.append(x)
#     llist1.append(y)
#     llist2.append((x,y))
# right=0
# left=0
# ssum=0
# ll_flag=0
# rr_flag=1
# while right<b or left<a:
#     mm=max(llist1)
#     temp=[]
#     # max最小值
#     for i in range(len(llist2)):
#         if mm==llist2[i][0] and left<a:
#             temp.append((llist2[i][1],i,ll_flag))
#         elif mm==llist2[i][1] and right<b:
#             temp.append((llist2[i][0],i,rr_flag))
#     sorted(temp)
#     # ssum+=temp[-1][1]
#     ssum+=mm
#     if temp[-1][2]==0:
#         left+=1
#     else:
#         right+=1
#     llist1.pop(mm)
#     llist1.pop(temp[-1][0])
#     llist2.pop(temp[-1][1])
# print(ssum)



#旅游产品
# import sys
# # sys.stdin = open('input.txt', 'r')
# llist= list(map(int, sys.stdin.readline().strip().split(' ')))
# a=int(sys.stdin.readline().strip())
# def computer(llist,num):
#     llist=sorted(llist,reverse=True)
#     def dfs(num,ll):
#         if num==0:
#             return 0
#         for i in ll:
#             if num-i>=0:
#                 return dfs(num-i,ll)+1
#             else:
#                 continue
#     return dfs(num,llist) if dfs(num,llist) else -1
# print(computer(llist,a))

# import sys
# sys.stdin = open('input.txt', 'r')
# m,n=map(int, sys.stdin.readline().strip().split(' '))
# llist=[[0 for i in range(m)] for t in range(m)]
# ll=[]
# for t in range(n):
#     a,b= list(map(int, sys.stdin.readline().strip().split(' ')))
#     llist[a][b]=1
#     ll.append((a,b))
# # print(llist[0][1])
# def Next(llist,node):
#     ll=[]
#     tt=llist[node[1]]
#     for i in range(len(tt)):
#         if tt[i]==1:
#             ll.append((node[0],i))
#     return ll


# def dfs(llist,start):
#     stack=[]
#     stack.append(start)
#     while len(stack)>0:
#         node=stack.pop()
#         if 
#         templ=Next(llist,node)
#         if templ:
#             stack+=templ
# for z in ll:
#     dfs(llist,z)

# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# #前序中序求叶子结点
# class Solution:
#     def reConstructBinaryTree(self, pre, tin):#pre、tin分别是前序序列和中序序列
#         # write code here
#         if len(pre)>0:
#             root=TreeNode(pre[0])#前序序列的第一个肯定是当前子树的根节点
#             rootid=tin.index(root.val)#通过根节点在中序序列中的位置划分出左右子树包含的节点
#             root.left=self.reConstructBinaryTree(pre[1:1+rootid],tin[:rootid])#重建左子树
#             root.right=self.reConstructBinaryTree(pre[1+rootid:],tin[rootid+1:])#重建右子树
#             return root
#     def levelOrder(self, root):
#         if not root:return []
#         stack=[]
#         stack.append(root)
#         res=[]
#         while len(stack)>0:
#             temp=[]
#             for i in range(len(stack)):
#                 node=stack.pop(0)
#                 temp.append(node.val)
#                 if node.left:
#                     stack.append(node.left)
#                 if node.right:
#                     stack.append(node.right)
#             res.append(temp[:])
#         return res
# import sys
# sys.stdin = open('input.txt', 'r')
# a=int(sys.stdin.readline().strip())
# pre=list(map(int, sys.stdin.readline().strip().split(' ')))
# tin=list(map(int, sys.stdin.readline().strip().split(' ')))
# SS=Solution()
# print(len(SS.levelOrder(SS.reConstructBinaryTree(pre,tin))[-1]))

#滑动窗口
# import sys
# sys.stdin = open('input.txt', 'r')
# a=int(sys.stdin.readline().strip())

# def slipwindous(llist,sstr):
#     needs=dict()
#     windous=dict()
#     for i in sstr:
#         if i in needs:
#             needs[i]=needs[i]+1
#         else:
#             needs[i]=1
#     right=0
#     left=0
#     void=0
#     res=[]
#     while right<len(llist):
#         c=llist[right]
#         right+=1
#         if c in windous:
#             windous[c]=windous[c]+1
#         else:
#             windous[c]=1
#         if c in needs and windous[c]==needs[c]:
#             void+=1
#         while void==len(needs):
#             d=llist[left]
#             left+=1
#             if right-left+1==len(sstr):
#                 res.append([left,right])
#             if d in needs:
#                 if windous[d]==needs[d]:
#                     void-=1
#                 windous[d]=windous[d]-1
#     return res
# def judge(intervals):
#         res = []
#         intervals.sort()
#         for i in intervals:
#             if not res or res[-1][1]<i[0]:
#                 res.append(i)
#             else:
#                 res[-1][1] = max(res[-1][1],i[1])
#         return res


# for i in range(a):
#     pre=sys.stdin.readline().strip().split(' ')[0]
#     print(len(judge(slipwindous(pre,"0010"))))

#第三题
#第四题
# import sys
# sys.stdin = open('input.txt', 'r')
# n,m=map(int, sys.stdin.readline().strip().split(' '))
# tin=list(map(int, sys.stdin.readline().strip().split(' ')))
# def computer(llist,lost):
#     def cc(ll):
#         ss=0
#         for i in ll:
#             ss+=llist[i]
#         return ss%lost
#     res=0
#     temp=[]
#     flag=0
#     def dfs(temp):
#         nonlocal flag
#         nonlocal res
#         res=max(cc(temp),res)
#         if len(temp)!=0 and temp[-1]==len(llist)-1:
#             return
#         while flag<len(llist):
#             for i in range(flag,len(llist)):
#                 if i not in temp:
#                     dfs(temp+[i])
#                     temp=[]
#                     flag+=1
#     dfs(temp+[0])
#     return res
# print(computer(tin,m))


#您好中国
# import sys
# sys.stdin = open('input.txt', 'r')
# def compute():
#     a=int(sys.stdin.readline().strip())
#     llist=[]
#     for i in range(a):
#         tin=list(sys.stdin.readline().strip())
#         llist.append(tin)
#     res=[]
#     mm="CHINA"
#     def dfs(x,y,t,temp):
#         # print(llist[x][y])
#         if t==len(mm)-1:
#             res.append(temp)
#         if not 0<=y<len(llist) or not 0<=x<len(llist[0]) or t==len(mm) or llist[x][y]!=mm[t]:
#             return
#         tmp,llist[x][y] = llist[x][y],'/'
#         dfs(x,y+1,t+1,temp+[tmp]) or dfs(x+1,y,t+1,temp+[tmp]) or dfs(x-1,y,t+1,temp+[tmp]) or dfs(x,y-1,t+1,temp+[tmp])
#         llist[x][y]=tmp
#         return res
#     for v in range(len(llist)):
#         for u in range(len(llist[0])):
#             dfs(v,u,0,[])
#     return res

# print(len(compute()))

# def qiuck_sort(llist,i,j):
#     def compute(left,right,pivot):
#         pivott=llist[pivot]
#         llist[right],llist[pivot]=llist[pivot],llist[right]
#         index=left
#         for i in range(left,right):
#             if llist[i]>pivott:
#                 llist[i],llist[index]=llist[index],llist[i]
#                 index+=1
#         llist[index],llist[right]=llist[right],llist[index]
#         return index
#     if i>=j:
#         return llist
#     pivot=i
#     low=i
#     high=j
#     index=compute(i,j,pivot)
#     qiuck_sort(llist,low,i-1)
#     qiuck_sort(llist,i+1,high)
#     return llist
# llist=[62,35,23,54,48,10]
# print(qiuck_sort(llist,0,len(llist)-1))

#归并排序
# def merge(a,b):
#     c=[]
#     right=left=0
#     while right<len(a) and left<len(b):
#         if a[right]<b[left]:
#             c.append(a[right])
#             right+=1
#         else:
#             c.append(b[left])
#             left+=1
#     if right==len(a):
#         for i in b[left:]:
#             c.append(i)
#         for t in a[right:]:
#             c.append(t)
#     return c

# def merge_sort(llist):
#     if len(llist)<=1:
#         return llist
#     mid=len(llist)//2
#     left=merge_sort(llist[:mid])
#     right=merge_sort(llist[mid:])
#     return merge(left,right)
# print(merge_sort([20,10,39,13,29,32]))

#快排
# def quick_sort(llist,i,j):
#     def patation(left,right,pivot):

#     if i>=j:
#         return llist
#     pivot=i
#     low=i
#     high=t
#     index=patation(i,j,pivot)
#     quick_sort(llist,i+1,high)
#     quick_sort(llist,low,i-1)
#     return llist

# lists=[62,35,23,54,48,10]
# print(quick_sort(lists,0,len(lists)-1))
# print("okok")

# def computer(nums,llist):
#         nums=[1,2,3]
#         res=0
#         n=2
#         def quan(nums,llist):
#                 print(llist)
#                 nonlocal res
#                 if sum(nums)%3==0:
#                         res+=1
#                         return
#                 for i in nums:
#                         if llist.length!=n:

#                                 llist.append(i)
#                                 print(llist)
#                                 quan(nums,llist)
#                                 llist.pop()
#                 return res
# nums=[2,3,5]
# print(computer(nums,[]))
# import sys
# n=int(sys.stdin.readline().strip())
# def main(n):
#     quene=[2,3,5]
#     flag=0
#     while len(quene)>0:
#         val=quene.pop(0)
#         flag+=1
#         if flag==n:
#             return val
#         quene.append(val*10+2)
#         quene.append(val*10+3)
#         quene.append(val*10+5)
# print(main(3))

#三角形的最小路径
# import sys
# sys.stdin = open('input.txt', 'r')
# n=int(sys.stdin.readline().strip())#行数
# triangle=[]
# for i in range(n):
#     temp=list(map(int, sys.stdin.readline().strip().split(' ')))
#     triangle.append(temp)
#     # print(triangle)
# class Solution:
#     def minimumTotal(self, triangle):
#         n=len(triangle)
#         dp=[[0 for t in range(n)] for i in range(n)]
#         dp[-1]=triangle[-1]
#         for z in range(n-1,-1,-1):
#             t=len(triangle[z])
#             for u in range(t-1):
#                 dp[z-1][u]=min(dp[z][u],dp[z][u+1])+triangle[z-1][u]
#         return dp[0][0]
# ss=Solution()
# print(ss.minimumTotal(triangle))

# def merge(a,b):
#     c=[]
#     i=0
#     t=0
#     while i<len(a) and t<len(b):
#         if a[i]>b[t]:
#             c.append(b[t])
#             t+=1
#         else:
#             c.append(a[i])
#             i+=1
#     if i==len(a):
#         for u in b[t:]:
#             c.append(u)
#     else:
#         for z in a[i:]:
#             c.append(z)
#     return c
# #归并
# def merge_sort(llist):
#     if len(llist)<2:
#         return llist
#     mid=len(llist)//2
#     left=merge_sort(llist[:mid])
#     right=merge_sort(llist[mid:])
#     return merge(left,right)
# print(merge_sort([3,42,36,1,3,78]))
# import sys
# import math
# sys.stdin = open('input.txt', 'r')
# m,n = map(int,sys.stdin.readline().strip().split(' '))
# temp=list(map(int, sys.stdin.readline().strip().split(' ')))
# n=int(sys.stdin.readline().strip())
# S=sys.stdin.readline().strip()
# T=sys.stdin.readline().strip()
# # print(len(S))
# def xiangsi(S,T):
#     def isSubsequence(s, t):#s为t的子序列
#         if not s:
#             return True
#         for i in t:
#             if s[0] == i:
#                 s = s[1:]
#             if not s:
#                 return True
#         return False
#     def addd(num):
#         sum=0
#         for i in range(num,-1,-1):
#             sum+=i
#         return sum
#     left=0
#     right=0
#     res=0
#     while right<=len(S):
#         if not isSubsequence(S[left:right],T) or right==len(S):
#             temp=right-left
#             res+=addd(temp)
#             left+=1
#         right+=1
#     return res
# print(xiangsi(S,T))

#再看看
# class Solution:
#     def merge(self , A, m, B, n):
#         right=n-1
#         left=m-1
#         cur=m+n-1
#         while right>-1 and left>-1:
#             if A[left]<B[right]:
#                 A[cur]=B[right]
#                 right-=1
#             else:
#                 A[cur]=A[left]
#                 left-=1
#             cur-=1
#         if right != -1: A[:right + 1] = B[:right + 1]
#         return A


# ss=Solution()
# print(ss.merge([1,2,3,0,0,0],3,[1,5,6],3))
# import sys
# sys.stdin = open('input.txt', 'r')
# temp=list(map(int, sys.stdin.readline().strip().split(' ')))
# n=int(sys.stdin.readline().strip())
# class Solution:
#     def GetMaxConsecutiveOnes(self , arr , k ):
#         # write code here
#         # ans=0
#         # count=0
#         # lo=0
#         # for i in range(len(arr)):
#         #     if arr[i]==0:
#         #         count+=1
#         #         while count>k:
#         #             if arr[0]==0:
#         #                 count-=1
#         #             lo+=1
#         #     ans=max(ans,i-lo+1)
#         # return ans
#         ans, count = 0, 0
#         flag= 0
#         for i in range(len(arr)):
#             if arr[i] == 0:
#                 count += 1
#                 while count > k:
#                     if arr[flag] == 0:
#                         count -= 1
#                     flag += 1
#             ans = max(ans, i - flag + 1)
#         return ans
# ss=Solution()
# print(ss.GetMaxConsecutiveOnes([1,1,1,0,0,0,1,1,1,1,0]
# ,2))

#         if not matrix:
#             return []
#         row,col=len(matrix),len(matrix[0])
#         i=0
#         j=-1
#         direction=1    #遍历的方向
#         res=[]
#         while row>0 and col>0:
#             #遍历行，处理列索引
#             for x in range(col):
#                 j+=direction
#                 res.append(matrix[i][j])
#             #遍历列 ，处理行索引
#             for y in range(row-1):  #由于前面已经遍历了一行，所以下面一行处理的时候，行需要减少一行
#                 i+=direction
#                 res.append(matrix[i][j])
#             direction=direction*-1
#             row,col=row-1,col-1
#         return res
# class Solution:
#     def GetFragment(self , str ):
#         sLen = len(str)
#         if sLen<2: 
#             return round(sLen,2)
#         result = []
#         temp = 1
#         for i in range(1,sLen):
#             if str[i]==str[i-1]:
#                 temp +=1
#             else:
#                 result.append(temp)
#                 temp = 1
#         result.append(temp) 
#         result =sum(result)//len(result) 
#         return result
# # 测试
# ss=Solution()
# s = "aaabbaaac"
# print(ss.GetFragment(s))

# import sys
# sys.stdin = open('input.txt', 'r')
# tt=input()
# zz=(tt[1:len(tt)-1])
# tmp = list(map(int, zz.split(',')))
# def jump(tmp):
#     mm_i=0
#     for i,n in enumerate(tmp):
#         if mm_i>=i and i+n>mm_i:
#             mm_i=i+n
#     print(mm_i>=i)
# jump(tmp)

# import sys
# sys.stdin = open('input.txt', 'r')
# ss=sys.stdin.readline().strip().split(" ")

# def find(ss):
#     length=len(ss)
#     for i in range(length):
#         strr=ss[i]
#         if not 8<=len(strr)<=120:
#             print(1)
#             continue
#         flag=[0 for i in range(4)]
#         string = "~!@#$%^&*()_+-*/<>,.[]\/"
#         for t in strr:
#             if t.isdigit():
#                 flag[0]+=1
#             elif t.isupper():
#                 flag[1]+=1
#             elif t.islower():
#                 flag[2]+=1
#             elif t in string:
#                 flag[3]+=1
#         if min(flag)>0:
#             print(0)
#         else:
#             print(2)
# find(ss)

# import sys
# sys.stdin = open('input.txt', 'r')
# word = sys.stdin.readline().strip().split(' ')
# class Solution:
#     def exist(self, board, word):
#         def dfs(i, j, k):
#             if not 0 <= i < len(board) or not 0 <= j < len(board[0]) or board[i][j] != word[k]: return False
#             if k == len(word) - 1: return True#满足结束条件返回
#             tmp, board[i][j] = board[i][j], '/'#做选择
#             res = dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1)
#             board[i][j] = tmp#撤回选择
#             return res

#         for i in range(len(board)):
#             for j in range(len(board[0])):
#                 if dfs(i, j, 0): return True
#         return False
# ss=Solution()
# board=[

#   ['A','B','C','E'],

#   ['S','F','C','S'],

#   ['A','D','E','E']

# ]
# print(ss.exist(board,word[0]))


# import sys
# import re
# sys.stdin = open('input.txt', 'r')
# no = sys.stdin.readline().strip()
# word = sys.stdin.readline().strip()
# tihuan= sys.stdin.readline().strip()

# def strfind(s,p,tihuan):
#     windows=dict()
#     needs=dict()
#     for i in p:
#         if i in needs:
#             needs[i]=needs[i]+1
#         else:
#             needs[i]=1
#     right=left=void=0
#     while right<=len(s)-1:
#         c=s[right]
#         right+=1
#         if c in needs:
#             if c in windows:
#                 windows[c]=windows[c]+1
#             else:
#                 windows[c]=1
#             if windows[c]==needs[c]:
#                 void+=1
#             while right-left>=len(p):
#                 if right-left==len(p) and void==len(needs):
#                     print(s[left:right])
#                     s=s[:left]+tihuan+s[right:]
#                     void=0
#                     for i in windows:
#                         windows[i]=0
#                     break
#                 left+=1

                    
#     return s
# print(strfind(word,no,tihuan))

# class Solution:
#     def permutation(self, s: str):
#         res=set()
        
#         def dfs(ss):
#             if len(ss)==len(s):
#                 temp=[]
#                 for i in ss:
#                     temp.append(s[i])
#                 res.add("".join(temp))
#                 return
#             for i in range(len(s)):
#                 if i not in ss:
#                     ss.append(i)
#                     dfs(ss)
#                     ss.pop()
#         dfs([])
#         return list(res)
# ss=Solution()
# print(ss.permutation("abc"))


# import sys
# sys.stdin = open('input.txt', 'r')
# word = sys.stdin.readline().strip().split(' ')
# def quan(llist):
#     # res=[]
#     def dfs(temp,res,resflag):
#         if len(llist)==len(temp):
#             print("".join(temp))
#             return
#         for i in range(len(llist)):
#             if i not in res:
#                 res.append(i)
#                 for u in range(resflag[-1],len(llist[i])):
#                     resflag.append(u)
#                     temp.append(llist[i][u])
#                     dfs(temp,res,resflag)
#                     temp.pop()
#                     res.pop()
#                     resflag.pop()
#     dfs([],[],[0])
# print(quan(word))                   

# class Solution:
#     def lastRemaining(self , n , m ):
#         names=[i for i in range(n)]
#         quene=[]
#         for t in names:
#             quene.append(t)
#         flagm=0
#         while len(quene)!=1:
#             node=quene.pop(0)
#             if flagm==m-1:
#                 flagm=0
#                 continue
#             quene.append(node)
#             flagm+=1
#         return quene[0]
# ss=Solution()
# ss.lastRemaining(3,2)

# import copy        
# class Solution:
#     def make_cancellation(self , content , bomb ):
#         # write code here
#         llist=list(map(str,content))
#         left=0
#         right=left+1
#         temp=-999
#         while right<len(content):
#             if llist[left]==llist[right]:
#                 temp=copy.deepcopy(llist[left])
#                 while right+1<len(content) and llist[right+1]==temp: 
#                     right+=1
#                 while left<right:
#                     llist[left]=""
#                     llist[right]=""
#             if str(temp)==bomb:
#                 if left-1>=0:
#                     # llist.pop(left-1)
#                     llist[left-1]=""
#                 if right+1<len(content):
#                     llist[right+1]=""
#             right+=1
#             left+=1
#         for z in range(len(llist)-1,-1,-1):
#             if llist[z]=='':
#                 llist.pop(z)
#             else:
#                 llist[z]=str(llist[z])
#         return "".join(llist) 

# ss=Solution()
# print(ss.make_cancellation("dsaffds11123aa","a"))

# import sys
# import re
# sys.stdin = open('input.txt', 'r')
# word = sys.stdin.readline().strip()
# ret=re.findall(r"[a-zA-Z0-9]+", word)
# ret=[i for i in ret if i!=""]
# temp=[]
# temp.append(ret[0].lower())
# for i in range(1,len(ret)):
#     tmp=ret[i][0].upper()+ret[i][1:].lower()
#     temp.append(tmp)
# print("".join(temp))

# import sys
# sys.stdin = open('input.txt', 'r')
# n, m = map(int, sys.stdin.readline().strip().split(' '))
# temp=[]
# for i in range(n):
#     tmp = list(map(int, sys.stdin.readline().strip().split(' ')))
#     temp.append(tmp)
# def compute(temp):
#     for t in range(len(temp)-1):
#         if temp[t]==temp[t+1]:
#             z=1
#             while t-z>0 and t+z+1<len(temp) and temp[t-z]==temp[t+z+1]:
#                 z+=1
#             if t-z==0 and temp[0]==temp[-1]:
#                 return temp[:z+1]
#             else:
#                 continue
#     return temp[:]
# tt=compute(temp)
# for u in tt:
#     for y in u:
#         print(y,end='')
#         print(" ",end="")
#     print()   

# import sys
# sys.stdin = open('input.txt', 'r')
# n, m,k = map(int, sys.stdin.readline().strip().split(' '))
# tmp = list(map(int, sys.stdin.readline().strip().split(' ')))
# ss=""
# for i in range(len(tmp)):
#     if tmp[i]>=k:
#         ss=ss+"1"
#     else:
#         ss=ss+" "
# ll=ss.split(" ")
# ss=0
# for i in ll:
#     if len(i)-m+1>0:
#         ss=ss+len(i)-m+1
# print(ss)

# import sys
# sys.stdin = open('input.txt', 'r')
# n, k,d = map(int, sys.stdin.readline().strip().split(' '))

# import sys
# sys.stdin = open('input.txt', 'r')
# n=int(sys.stdin.readline().strip())
# temp=[]
# def cmpd(ll):
#     return ll[1],ll[0]
# for i in range(n):
#     a,b= map(int, sys.stdin.readline().strip().split(' '))
#     temp.append([i+1,a,b])
# tt=sorted(temp,key=lambda x:[x[2],x[1]],reverse=True)
# for i in range(len(tt)):
#     print(tt[i][0],end="")
#     print(" ",end="")
# students = [[3,'Jack',12],[2,'Rose',13],[1,'Tom',10],[5,'Sam',12],[4,'Joy',8]]
# ss=sorted(students,key=(lambda x:x[2]))
# print(ss)

# import sys
# sys.stdin = open('input.txt', 'r')
# n, k,d = map(int, sys.stdin.readline().strip().split(' '))
# import itertools
# list1 = [1,2,1]
# list2 = []
# for i in range(1,len(list1)+1):
#     iter = itertools.permutations(list1,i)
#     list2.append(list(iter))
# print(list2)

#n,k,d。z个数相加等于k，最大数大于d,别的数在1～n之间，求有几种方法。
# import sys
# sys.stdin = open('input.txt', 'r')
# n, k, d = map(int, input().split())
# dp_k = [1]
# d -= 1
# dp_d = [1]
# for i in range(1,n):
#     start_k = max(0,i-k)
#     start_d = max(0,i-d)
#     tmp_k = 1 if i < k else 0
#     tmp_d = 1 if i < d else 0
#     dp_k.append((sum(dp_k[start_k:i])+tmp_k) % 998244353)
#     dp_d.append((sum(dp_d[start_d:i])+tmp_d) % 998244353)
# print((dp_k[-1] - dp_d[-1] + 998244353) % 998244353)

# import sys
# sys.stdin = open('input.txt', 'r')
# n,k,d = map(int,input().split())
# dp = [0 for i in range(n+1)]
# dp[0] = 1
# dp2 = [0 for i in range(n+1)]
# dp2[0] = 1
# for i in range(1,n+1):
#     for j in range(1,k+1):
#         if i - j >= 0:
#             dp[i] += dp[i-j]
# for i in range(1,n+1):
#     for j in range(1,d):
#         if i - j >= 0:
#             dp2[i] += dp2[i-j]
# print((dp[n] - dp2[n]) % 998244353)

# def coinsget(coins,amount):
#     dp = [amount + 1 for _ in range(amount + 1)]
#     dp[0] = 0
#     for i in range(1, amount + 1):
#         for coin in coins:
#             if coin <= i and dp[i - coin] != amount + 1:
#                 dp[i] = min(dp[i], dp[i-coin] + 1)
#     if dp[amount] == amount + 1:
#         dp[amount] = -1
#     return dp[amount]

# print(coinsget([1,2,5],11))

# import sys
# sys.stdin = open('input.txt', 'r')
# n=int(sys.stdin.readline().strip())
# tmp = list(map(str, sys.stdin.readline().strip()))
# # print(tmp)
# for i in range(0,len(tmp),n):
#     temp=tmp[i:i+n]
#     temp.reverse()
#     tmp[i:i+n]=temp
# print("".join(tmp))
# import re
# s=input()
# regex_start = re.compile("*@*.com")
# print(regex_start.findall(s))506

# class Linklist:
#     def __init__(self, value):
#         self.next=None
#         self.v=value

# class Solution:
#     def hasCycle(self , head ):
#         # write code here
#         if head==None:
#             return False
#         slow=head
#         fast=head
#         while fast.next is not None and fast is not None:
#             slow=slow.next
#             fast=fast.next.next
#             if slow==fast:
#                 return True
#         return False
# tt=Linklist(1)
# tt.next=tt

# ss=Solution()
# print(ss.hasCycle(tt))
# import sys
# sys.stdin = open('input.txt', 'r')
# ss=list(map(int,sys.stdin.readline().strip().split()))
#1.判断字符串是不是全是小写
# import sys
# import re
# # pattern = re.compile(r'hello')
# m = re.findall(r'[0-9a-zA-Z]{0,19}@[\d\w]+.com', '50603929@*.com')
# print(m.group())
# from random import randint
# class Solution:
#     def __init__(self,dd):
#         self.llist=[]
#         for i,v in dd.items():
#             for t in range(v):
#                 self.llist.append(i)
#     def get(self):
#         return self.llist[randint(0,len(self.llist)-1)]
# dd=dict()
# dd['A']=1
# dd['B']=2
# dd['C']=3
# ss=Solution(dd)
# for t in range(60):
#     print(ss.get())
# class Solution:
#     def merge(self , A, m, B, n):
#         if not A and not B: return []
#         elif not A: return B
#         elif not B: return A
#         right=n-1
#         left=m-1
#         cur=m+n-1
#         while right>=0 and left>=0:
#             if A[right]>B[left]:
#                 A[cur]=A[right]
#                 right-=1
#             else:
#                 A[cur]=B[left]
#                 left-=1
#             cur-=1
#         if right>0:
#             A[:n] = B[:n]
#         return A
# ss=Solution()
# print(ss.merge([0],0,[1],1))

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# def merge(lists):
#     if not lists: return 
#     result = []
#     for list1 in lists:
#         while list1:
#             result.append(list1.val)
#             list1 = list1.next
#     result.sort()
#     point = head = ListNode(0)
#     for i in result:
#         point.next = ListNode(i)
#         point = point.next 
#     return head.next
# import sys
# sys.stdin = open('input.txt', 'r')
# a=int(sys.stdin.readline().strip())
# lists=[]
# for i in range(a):
#     ss=sys.stdin.readline()
#     if len(ss)>2:
#         tmp= list(map(int, ss.strip().split(' ')))
#         nodef=ListNode(-1)
#         nodetmp=nodef
#         for i in range(len(tmp)):
#             nodef.next=ListNode(tmp[i])
#             nodef=nodef.next
#         lists.append(nodetmp.next)   
# knode=merge(lists)
# while knode:
#     print(knode.val,end=" ")
#     knode=knode.next
# print()
    

# import sys
# #sys.stdin = open('input.txt', 'r')
# class Solution(object):
#     def searchRange(self, nums, target):
#         loww, higg = 0, len(nums) - 1
#         while(loww <= higg):
#             mid = (loww + higg) // 2
#             if nums[mid] == target:
#                 break
#             elif nums[mid] > target:
#                 higg = mid - 1
#             else:
#                 loww = mid + 1
#         if loww > higg:
#             return [-1, -1]
#         midtarget = mid
#         loww, higg = 0, mid
#         lefpos = 0
#         while(loww <= higg):
#             if (higg >= 1 and nums[higg - 1] != target) or higg == 0: 
#                 lefpos = higg
#                 break
#             mid = (loww + higg) // 2
#             if nums[mid] == target:
#                 higg = mid
#             elif nums[mid] < target:
#                 loww = mid + 1
#         rigtpos = 0
#         loww, higg = midtarget, len(nums) - 1
#         while(loww <= higg):
#             if (loww <= len(nums) - 2 and nums[loww + 1] != target) or loww == len(nums) - 1:
#                 rigtpos = loww
#                 break
#             mid = (loww + higg + 1) // 2
#             if nums[mid] == target:
#                 loww = mid 
#             elif nums[mid] > target:
#                 higg = mid - 1
#         return lefpos, rigtpos
# ss=Solution()
# m, n = map(int, sys.stdin.readline().strip().split(' '))
# tmp = list(map(int, sys.stdin.readline().strip().split(' ')))
# a,b=ss.searchRange(tmp,n)
# print(a,b)

# class TrieNode:
#     def __init__(self):
#         self.nodes = dict()  # 构建字典
#         self.is_leaf = False
#     def insert(self, word: str): 
#         curr = self
#         for char in word:
#             if char not in curr.nodes:
#                 curr.nodes[char] = TrieNode()
#             curr = curr.nodes[char]
#         curr.is_leaf = True
#     def insert_many(self, words: [str]):
#         for word in words:
#             self.insert(word)
#     def search(self, word: str):
#         curr = self
#         for char in word:
#             if char not in curr.nodes:
#                 return False
#             curr = curr.nodes[char]
#         return curr.is_leaf
# class MyTopK:
#     def topK(self,k: int, nums: list):
#         heap = self.buidHeap(nums[0:k])
#         for i in range(k,len(nums)):
#             if nums[i] > heap[0]:
#                 heap[0] = nums[i]
#                 heap[0],heap[-1] = heap[-1], heap[0]
#                 heap = self.buidHeap(heap)
#         return heap
 
#     def buidHeap(self,heap:list):
#         # 数组中的数据是按照二叉树 层次遍历的，自下而上,遍历所有根节点， 将最小值放入根节点
#         heap_len = len(heap)
#         for i in range(heap_len // 2,-1,-1):
#             left, right = 2*i + 1,2*i + 2
#             if left < heap_len and heap[left] < heap[i]:
#                 heap[left], heap[i] = heap[i], heap[left]
#             if right < heap_len and heap[right] < heap[i]:
#                 heap[right], heap[i] = heap[i], heap[right]
#         return heap
# if __name__ == '__main__':
#     s = MyTopK().topK(5,[2,3,6,1,8,23,56,78,12])
#     print(s)

# class Solution:
#     def topKFrequent(self, nums, k):
#         dic = {}
#         for num in nums:# 统计个数
#             if num in dic:
#                 dic[num] += 1
#             else:
#                 dic[num] = 1
        
#         def Quick_sort(arr,dic): # 定义堆排序
#             length = len(arr)
#             k = length >> 1
#             for i in reversed(range(k)):
#                 sink(i,length,arr,dic)
        
#         def sink(n,length,arr,dic): # 堆排序核心
#             left = 2*n + 1
#             right = 2*n + 2
#             if left >= length:
#                 return 
#             min_ = left
#             if right < length and dic[arr[left]] > dic[arr[right]]:
#                 min_ = right
#             if dic[arr[n]] > dic[arr[min_]]:
#                 arr[n],arr[min_] = arr[min_], arr[n]
#                 sink(min_,length,arr,dic)
        
#         arr = list(set(nums)) # set()去重，时间复杂度为O(n)
#         temp = arr[:k]
#         Quick_sort(temp,dic)
#         for num in arr[k:]:
#             if dic[num] > dic[temp[0]]:
#                 temp[0] = num
#                 sink(0,k,temp,dic)# 调整堆，将频率最低的调整到堆头
        
#         return temp

# ss=Solution()
# print(ss.topKFrequent([1,1,1,2,2,3],1))

# class DLinkedNode:
#     def __init__(self, key=0, value=0):
#         self.key = key
#         self.value = value
#         self.prev = None
#         self.next = None
# class LRUCache:
#     def __init__(self, capacity: int):
#         self.cache = dict()
#         # 使用伪头部和伪尾部节点    
#         self.head = DLinkedNode()
#         self.tail = DLinkedNode()
#         self.head.next = self.tail
#         self.tail.prev = self.head
#         self.capacity = capacity
#         self.size = 0
#     def get(self, key: int) -> int:
#         if key not in self.cache:
#             return -1
#         # 如果 key 存在，先通过哈希表定位，再移到头部
#         node = self.cache[key]
#         self.moveToHead(node)
#         return node.value

#     def put(self, key: int, value: int) -> None:
#         if key not in self.cache:
#             # 如果 key 不存在，创建一个新的节点
#             node = DLinkedNode(key, value)
#             # 添加进哈希表
#             self.cache[key] = node
#             # 添加至双向链表的头部
#             self.addToHead(node)
#             self.size += 1
#             if self.size > self.capacity:
#                 # 如果超出容量，删除双向链表的尾部节点
#                 removed = self.removeTail()
#                 # 删除哈希表中对应的项
#                 self.cache.pop(removed.key)
#                 self.size -= 1
#         else:
#             # 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
#             node = self.cache[key]
#             node.value = value
#             self.moveToHead(node)
#     def addToHead(self, node):
#         node.prev = self.head
#         node.next = self.head.next
#         self.head.next.prev = node
#         self.head.next = node
    
#     def removeNode(self, node):
#         node.prev.next = node.next
#         node.next.prev = node.prev

#     def moveToHead(self, node):
#         self.removeNode(node)
#         self.addToHead(node)

#     def removeTail(self):
#         node = self.tail.prev
#         self.removeNode(node)
#         return node
#     def printf(self):
#         for k,v in self.cache.items():
#             print(k,v.value)
# ss=LRUCache(3)
# ss.put(1,2)
# ss.put(2,3)
# ss.put(3,4)
# ss.put(4,5)
# # ss.printf()
# print(ss.get(3))
# ss.put(6,7)
# ss.put(7,8)
# ss.printf()
# class LRUCache:
#     def __init__(self, capacity: int):
#         self.cache = {}
#         self.List = []
#         self.capacity = capacity

#     def get(self, key: int) -> int:
#         if key in self.cache:
#             self.List.remove(key)
#             self.List.append(key)
#             return self.cache[key]
#         else :
#             return -1


#     def put(self, key: int, value: int) -> None:
#         if len(self.cache) < self.capacity:
#             if key in self.cache:
#                 self.cache[key] = value
#                 self.List.remove(key)
#                 self.List.append(key)
#             else:
#                 self.cache[key] = value
#                 self.List.append(key)
#         else:
#             if key in self.cache:
#                 self.cache[key] = value
#                 self.List.remove(key)
#                 self.List.append(key)
#             else:
#                 self.cache.pop(self.List[0])
#                 self.List.remove(self.List[0])
#                 self.cache[key] = value
#                 self.List.append(key)
#     def printf(self):
#         for k,v in self.cache.items():
#             print(k,v)
# ss=LRUCache(3)
# ss.put(1,2)
# ss.put(2,3)
# ss.put(3,4)
# ss.put(4,5)
# # ss.printf()
# print(ss.get(3))
# ss.put(6,7)
# ss.put(7,8)
# ss.printf()



#encoding=utf-8
# import threading
# import time
 
# #python2中
# # from Queue import Queue
 
# #python3中
# from queue import Queue
 
# class Producer(threading.Thread):
#     def run(self):
#         global queue
#         count = 0
#         while True:
#             if queue.qsize() < 1000:
#                 for i in range(100):
#                     count = count +1
#                     msg = '生成产品'+str(count)
#                     queue.put(msg)
#                     print(msg)
#             time.sleep(0.5)
 
# class Consumer(threading.Thread):
#     def run(self):
#         global queue
#         while True:
#             if queue.qsize() > 100:
#                 for i in range(3):
#                     msg = self.name + '消费了 '+queue.get()
#                     print(msg)
#             time.sleep(1)
 
 
# if __name__ == '__main__':
#     queue = Queue()
 
#     for i in range(500):
#         queue.put('初始产品'+str(i))
#     for i in range(2):
#         p = Producer()
#         p.start()
#     for i in range(5):
#         c = Consumer()
#         c.start()
# class Solution:
#     def FindGreatestSumOfSubArray(self, array):
#             max_sum = array[0]
#             pre_sum = 0
#             for i in array:
#                 if pre_sum < 0:
#                     pre_sum = i
#                 else:
#                     pre_sum += i
#                 if pre_sum > max_sum:
#                     max_sum =  pre_sum
#             return max_sum

# def main():
#     lists=[6,-3,-2,7,-15,1,2,2]
#     print(function(lists))
    
# if __name__ == "__main__":
#     main()
# def compute(llist):
#     res=[]
#     def quan(temp):
#         if len(llist)==len(temp):
#             res.append(temp[:])
#             return 
#         for i in llist:
#             if i in temp:
#                 continue
#             temp.append(i)
#             quan(temp)
#             temp.pop()
#         return res
#     return quan([])

# print(compute([1,2,3]))
# def strorder(s1,s2):
#     need=dict()
#     windows=dict()
#     for i in s1:
#         if i in need:
#             need[i]=need[i]+1
#         else:
#             need[i]=1
#     right=0
#     left=0
#     vaild=0
#     while right<len(s2):
#         c=s2[right]
#         right+=1
#         if c in need:
#             if c in windows:
#                 windows[c]=windows[c]+1
#             else:
#                 windows[c]=1
#             if windows[c]==need[c]:
#                 vaild+=1
#         while right-left>=len(s1):
#             if vaild==len(need):
#                 return True
#             d=s2[left]
#             left+=1
#             if d in need:
#                 if windows[d]==need[d]:
#                     vaild-=1
#                 windows[d]=windows[d]-1
#     return False
# print(strorder("ct","abccecdt"))

# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = "ABCCED"
# def solution(board,word):
#     w=len(board)
#     h=len(board[0])
#     def compute(i,t,count):
#         if i>=w or t>=h or board[i][t]!=word[count]:
#             # print("no: "+board[i][t])
#             return False
#         if count==len(word)-1:
#             return True
#         board[i][t],temp='/',board[i][t]
#         flag=compute(i+1,t,count+1) or compute(i,t+1,count+1) or compute(i-1,t,count+1) or compute(i,t-1,count+1)
#         board[i][t]=temp
#         return flag
#     for i in range(len(board)):
#         for t in range(len(board[i])):
#             if compute(i,t,0):
#                 return True
#                 # print(board[i][t])
#     return False
# print(solution(board,word))

#黄金矿工
# class Solution:
#     def getMaximumGold(self, grid):
#         m=len(grid)#行
#         n=len(grid[0])#列
#         # allsum=0 我们缺少好的办法解决在递归中，变量恢复的问题。我们可以采用全局静态变量，问题是这种变量又不能在我们后续调用中起到归0的作用
#         # global maxsum
#         def compute(i,t,allsum):
#             nonlocal maxsum
#             if i>=m or t>=n or grid[i][t]==0:
#                 return 0
#             if allsum+grid[i][t]>maxsum:
#                 maxsum=allsum+grid[i][t]
#             temp,grid[i][t]=grid[i][t],0
#             compute(i+1,t,allsum)
#             compute(i,t+1,allsum)
#             compute(i-1,t,allsum)
#             compute(i,t-1,allsum)
#             grid[i][t]=temp
#             return allsum
#         maxsum=0
#         for i in range(m):
#             for t in range(n):
#                 maxsum=max(maxsum,maxsum+compute(i,t,0))
#         return maxsum
# ss=Solution()
# grid=[[0,6,0],[5,8,7],[0,9,0]]
# print(ss.getMaximumGold(grid))

#黄金矿工
# def getMaximumGold(grid):
#     m=len(grid)#行
#     n=len(grid[0])#列
#     # allsum=0
#     # maxsum=0
#     maxsum=[0]
#     def compute(i,t,allsum):
#         if i>=m or t>=n or grid[i][t]==0:
#             return 0
#         # print(maxsum)
#         if sum(allsum)>0:
#             maxsum.append(allsum)
#         temp,grid[i][t]=grid[i][t],0
#         compute(i+1,t,allsum+grid[i][t])
#         allsum.pop()
#         compute(i,t+1,allsum+grid[i][t])
#         allsum.pop()
#         compute(i-1,t,allsum+grid[i][t])
#         allsum.pop()
#         compute(i,t-1,allsum+grid[i][t])
#         allsum.pop()
#         grid[i][t]=temp
#         return maxsum
        
#     for i in range(m):
#         for t in range(n):
#             compute(i,t,[grid[i][t]])
#     return sum(maxsum)
# grid=[[0,6,0],[5,8,7],[0,9,0]]
# print(getMaximumGold(grid))

# class Solution:
#     def getMaximumGold(self, grid):
#         def dfs(grid,x,y,res):
#             nonlocal allgrid
#             if not 0<=x<len(grid) or not 0<=y<len(grid[0]) or grid[x][y]==0:return False
#             res.append(grid[x][y])
#             grid[x][y],temp=0,grid[x][y]
#             allgrid=max(allgrid,sum(res))
#             dfs(grid,x+1,y,res) or dfs(grid,x-1,y,res) or dfs(grid,x,y-1,res) or dfs(grid,x,y+1,res)
#             grid[x][y]=temp
#             res.pop()
#         allgrid=0
#         for i in range(len(grid)):
#             for t in range(len(grid[0])):
#                 dfs(grid,i,t,[])
#         return allgrid
# ss=Solution()
# print(ss.getMaximumGold([[0,6,0],[5,8,7],[0,9,0]]))

# class Solution:
#     def countNumbersWithUniqueDigits(self, n: int) -> int:
#         def judge(su):
#             if len(su)==0:return False

#             jj=[]
#             for t in su:
#                 if t in jj:
#                     return True
#                 jj.append(t)
#             return False
#         def compute(res):
#             nonlocal sum
#             if len(res)>n or judge(res):
#                 return
#             sum+=1
#             if len(res)==n:
#                 return 
#             for i in range(n):
#                 for t in range(9):
#                     res.append(t)
#                     compute(res)
#                     res.pop()
#         sum=0
#         compute([])
#         return sum

# ss=Solution()
# print(ss.countNumbersWithUniqueDigits(2))
# summ=0
# def compute(n):
#     global summ
#     if n==2:
#         summ=summ+1
#         return
#     if n>2:#这个点非常关键
#         if n%2==0:
#             summ=summ+n//2
#             compute(n//2)
#         else:
#             summ=summ+(n-1)//2
#             compute((n-1)//2+1)
#     return summ
# print(compute(7))
# func(8)

# class Solution:
#     def getHappyString(self, n,k) -> str:
#         maxtmp=0
#         ss=["a","b","c"]
#         res=""
#         def compute(tmp):
#             nonlocal maxtmp
#             nonlocal res
#             if len(tmp)>1 and tmp[-1] ==tmp[-2]:
#                 return
#             if len(tmp)==n:
#                 maxtmp+=1
#                 if maxtmp==k:
#                     res+=tmp
#                 return

#             for i in ss:

#                 tmp=tmp+i
#                 compute(tmp)

#                 tmp=tmp[:-1]
#         compute("")
#         return res

# ss=Solution()
# print(ss.getHappyString(1,4))


# class Solution:
#     def partition(self, s):
#         def recall(s, tmp):
#             if not s:
#                 res.append(tmp[:])
#                 return

#             for i in range(1, len(s)+1):
#                 if s[:i] == s[:i][::-1]:#反转字符串
#                     tmp.append(s[:i])
#                     recall(s[i:], tmp)
#                     tmp.pop()
        
#         res = []
#         recall(s, [])
#         return res
# ss=Solution()
# print(ss.partition("aaba"))

# class Solution:
#     def splitIntoFibonacci(self, S):
#         res=[]
#         def fb(llist):
#             if len(llist)<3:
#                 return False
#             for i in range(len(llist)-1):
#                 if i+2<len(llist) and int(llist[i])+int(llist[i+1])!=int(llist[i+2]):
#                     return False
#             return True

#         def compute(S,tmp):
#             if not fb(tmp) and len(tmp)>2:
#                 return
#             elif fb(tmp) and len(S)==0:
#                 temp=[]
#                 for i in tmp:
#                     temp.append(int(i))
#                 res.append(temp)
#                 return 
#             for l in range(1,len(S)+1):#这个+1操作很关键，如果不加一，只有一个数的字符串“1”无法被添加到tmp里面
#                 tmp.append(S[:l])
#                 compute(S[l:],tmp)
#                 tmp.pop()
#         compute(S,[])
#         return res[0]
# ss=Solution()
# S="1101111"
# print(ss.splitIntoFibonacci(S))

# class Solution:
#     def letterCombinations(self, digits: str) -> List[str]:

# def compute(llist):
#     tmp=[]    #tmp用于记录符合条件的结果
#     def sum(temp):#temp用于递归过程中进行记录枚举结果
#         if len(temp)==len(llist):#判断
#             tmp.append(temp[:])
#             return
#         for i in llist: #进行枚举
#             temp.append(i) #加入枚举变量
#             sum(temp) #递归判断
#             temp.pop() #剔除枚举变量
#     sum([])
#     return tmp
# print(compute([1,2,3]))

# def compute(llist):
#     tmp=[]    #tmp用于记录符合条件的结果
#     def summ(st,temp):#temp用于递归过程中进行记录枚举结果
#         if len(temp)==len(llist):#判断
#             tmp.append(temp[:])
#             return
#         for i in st: #进行枚举
            
#             temp.append(i) #加入枚举变量
#             summ(st[:i]+st[i+1:],temp) #递归判断
#             temp.pop() #剔除枚举变量
#     summ(llist,[])
#     return tmp
# print(compute([1,2,3]))


# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = "ABCCEC"
# def solution(board,word):
#     w=len(board)
#     h=len(board[0])
#     def compute(i,t,count):
#         if i>=w or t>=h or board[i][t]!=word[count]:
#             # print("no: "+board[i][t])
#             return False
#         if count==len(word)-1:
#             return True
#         board[i][t],temp='/',board[i][t]
#         flag=compute(i+1,t,count+1) or compute(i,t+1,count+1) or compute(i-1,t,count+1) or compute(i,t-1,count+1)
#         board[i][t]=temp
#         return flag
#     for i in range(len(board)):
#         for t in range(len(board[i])):
#             if compute(i,t,0):
#                 return True
#                 # print(board[i][t])
#     return False
# print(solution(board,word))

# class Solution:
#     def partition(self, s):
#         def recall(s, tmp):
#             if not s:
#                 res.append(tmp[:])
#                 return
#             for i in range(1, len(s)+1):
#                 if s[:i] == s[:i][::-1]:#反转字符串
#                     tmp.append(s[:i])
#                     recall(s[i:], tmp)
#                     tmp.pop()
        
#         res = []
#         recall(s, [])
#         return res
# ss=Solution()
# print(ss.partition("aaba"))

# def compute(nums):
#     tmp=[]
#     def sum(st,temp):
#         if len(temp)==len(nums):
#             tmp.append(temp[:])
#             return
#         seen=set()
#         for i in range(len(st)):
#             if st[i] in seen:
#                 continue
#             seen.add(st[i])
#             temp.append(st[i])
#             sum(st[:i]+st[i+1:],temp)
#             temp.pop()
#     sum(nums,[])
#     return tmp
# print(compute([1,2,3,4]))
# nums=[1,2,3,4]
# def quan(nums):
#     stack=[]
#     for t in nums:
#         stack.append(t)
#     res=[]
#     temp=[]
#     while stack:
#         node=stack.pop()
#         temp.append(node)
#         if node and len(temp)!=len(nums):
#             for i in nums:
#                 stack.append(i)
#         if len(temp)==len(nums):
#             res.append(temp[:])
#             temp.pop()
#             if node==nums[0]: 
#                 tnode=temp.pop()
#                 while temp and tnode==nums[0]:
#                     tnode=temp.pop()
#     return res
# print(quan(nums))

#最长子序列
# def longstr(nums):
#     dp=[1 for i in range(len(nums))]
#     for t in range(len(nums)):
#         for z in range(t):
#             if nums[t]>nums[z]:
#                 dp[t]=max(dp[t],dp[z]+1)
#     return max(dp)

# print(longstr([10,9,25,37,101,18]))
                

# class Solution:
#     def lengthOfLIS(self, nums):
#         self.length=0
#         def dfs(lists,t):
#             if t>=len(nums) or  len(lists)>1 and nums[t]<lists[-2]:
#                 return 
#             self.length=max(self.length,len(lists))
#             for i in range(t,len(nums)):
#                 node=nums[i]
#                 if node not in lists:
#                     lists.append(node)
#                     dfs(lists,i)
#                     lists.pop()
#         for t in range(len(nums)):
#             dfs([nums[t]],t)
#         return self.length
# ss=Solution()
# print(ss.lengthOfLIS([10,9,25,37,101,18]))

# def longstr(nums):
#     seen=[]
#     maxx=0
#     def compute(llist):
#         nonlocal maxx
#         if len(llist)>1 and llist[-1]<llist[-2]:
#             return
#         maxx=max(maxx,len(llist))
#         for i in range(len(nums)):
#             if i in seen:
#                 continue
#             seen.append(i)
#             llist.append(nums[i])
#             compute(llist)
#             llist.pop()
#         return maxx
#     return compute([])
# print(longstr([10,9,25,37,101,18]))

# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = "ABCC"
# def solution(board,word):
#     w=len(board)
#     h=len(board[0])
#     def compute(i,t,count):
#         if i>=w or t>=h or board[i][t]!=word[count]:
#             # print("no: "+board[i][t])
#             return False
#         if count==len(word)-1:
#             return True
#         board[i][t],temp='/',board[i][t]
#         flag=compute(i+1,t,count+1) or compute(i,t+1,count+1) or compute(i-1,t,count+1) or compute(i,t-1,count+1)
#         board[i][t]=temp
#         return flag
#     for i in range(len(board)):
#         for t in range(len(board[i])):
#             if compute(i,t,0):
#                 return True
#                 # print(board[i][t])
#     return False
# print(solution(board,word))

# 俄罗斯套娃信封
# class Solution:
#     def maxEnvelopes(self, envelopes):
#         if not envelopes:
#             return 0
#         envelopes.sort(key=lambda x: (x[0], -x[1]))
#         dp=[1 for i in range(len(envelopes))]

#         for t in range(len(envelopes)):
#             for z in range(t):
#                 if envelopes[t][1]>envelopes[z][1] and envelopes[t][0]>envelopes[z][0]:
#                     dp[t]=max(dp[z]+1,dp[t])
#         return max(dp)

# ss=Solution()
# print(ss.maxEnvelopes([[4,5],[4,6],[6,7],[2,3],[1,1]]))

#全排列动态规划:https://leetcode-cn.com/problems/permutations/solution/dong-tai-gui-hua-si-lu-jian-ji-ming-liao-by-bu-hui/
# class Solution:
#     def permute(self, nums):
#         if not nums:
#             return None
#         dp = [[] for i in range(len(nums))]
#         dp[0] = [[nums[0]]]
#         for i in range(1, len(nums)):
#             for ans in dp[i-1]:
#                 dp[i].append(ans+[nums[i]])
#                 for j in range(len(ans)):
#                     dp[i].append(ans[:j]+[nums[i]]+ans[j+1:]+[ans[j]])
#         return dp[-1]

# ss=Solution()
# print(ss.permute([1,2,3,4]))

# def partition(s):
#     def compute()
#         for i in range(1,len(s)):
#             temp=s[:i]
#             if temp==temp[::-1]:
#                 tmp.append(temp)#不理解
#                 compute(temp[i:],tmp)
#                 tmp.pop()
        



# print(partition("aaba"))

# def longstr(nums):
#     seen=[]
#     maxx=0
#     def compute(llist):
#         nonlocal maxx
#         if len(llist)>1 and llist[-1]<llist[-2]:
#             return
#         maxx=max(maxx,len(llist))
#         for i in range(len(nums)):
#             if i in seen:
#                 continue
#             seen.append(i)
#             llist.append(nums[i])
#             compute(llist)
#             llist.pop()
#         return maxx
#     return compute([])
# print(longstr([10,9,101,10,17,18]))


# class Solution:
#     def permute(self, nums):
#         if not nums:
#             return None
#         dp = [[] for i in range(len(nums))]
#         dp[0] = [[nums[0]]]
#         for i in range(1, len(nums)):
#             for ans in dp[i-1]:
#                 dp[i].append(ans+[nums[i]])
#                 for j in range(len(ans)):
#                     dp[i].append(ans[:j]+[nums[i]]+ans[j+1:]+[ans[j]])
#         return dp[-1]

# ss=Solution()
# print(ss.permute([1,2,3,4]))

# def longstr(nums):
#     dp=[1 for i in range(len(nums))]
#     for t in range(len(nums)):
#         for z in range(t):
#             if nums[t]>nums[z]:
#                 dp[t]=max(dp[t],dp[z]+1)
#     return max(dp)

# print(longstr([10,9,25,37,101,18]))

# def longzistr(nums):
#     dp=nums.copy()
#     for t in range(len(nums)):
#         for z in range(t):
#             dp[t]=max(nums[t],nums[t]+dp[z])
#     return max(dp)
# nums=[-10,10,-5,2,3]
# print(longzistr(nums))

# C=[10,10,5,2,3]
# S=[-1,1,-1,1,1]
# def compute(C,S):
#     nums = list(map(lambda x,y:x*y,C,S))
#     dp0=nums[0]
#     temp=0
#     res=0
#     for t in range(1,len(nums)):
#         temp=max(nums[t],nums[t]+dp0)
#         dp0=temp
#         res=max(res,temp)
#     return res
# print(compute(C,S))

# def longeststr(ss):
#     ls=len(ss)
#     dp=[[0 for i in range(ls)] for t in range(ls)]
#     for z in range(ls):
#         dp[z][z]=1
#     for i in range(ls-1,-1,-1):
#         for j in range(i+1,ls):
#             if ss[i]==ss[j]:
#                 dp[i][j]= dp[i+1][j-1]+2
#             else:
#                 dp[i][j]=max(dp[i+1][j],dp[i][j-1])
#     return dp[0][ls-1]

# print(longeststr("bbbab"))

# def longeststr(ss):
#     ls=len(ss)
#     dp=[1 for i in range(ls)]
#     for i in range(ls-2,-1,-1):
#         pre=0
#         for j in range(i+1,ls):
#             temp=dp[j]
#             if ss[i]==ss[j]:
#                 dp[j]=pre+2
#             else:
#                 dp[j]=max(dp[j],dp[j-1])
#             pre=temp
#     return dp[ls-1]
# print(longeststr("bbbab"))

# def onezerobag(N,W,wt,val):
#     dp=[[0 for i in range(W+1)] for i in range(N+1)]
#     for i in range(1,N+1):
#         for t in range(1,W+1):
#             if t-wt[i-1]<0:
#                 dp[i][t]=dp[i-1][t]
#             else:
#                 dp[i][t]=max(dp[i-1][t],dp[i-1][t-wt[i-1]]+val[i-1])
#     return dp[N][W]
# print(onezerobag(3,4,[2,1,3],[4,2,3]))



# def findMaxForm(strs, m: int, n: int):
#     if len(strs) == 0:
#         return 0
        
#     dp = [[0]*(n+1) for _ in range(m+1)]   #准备很多个背包
        
#     for strs_item in strs:
#         item_count0 = strs_item.count('0')
#         item_count1 = strs_item.count('1')
            
#             #遍历可容纳的背包 
#         for i in range(m, item_count0 - 1, -1):  #采取倒序
#             for j in range(n, item_count1 - 1, -1):
#                 dp[i][j] = max(dp[i][j], 1 + dp[i-item_count0][j-item_count1])
                    
#     return dp[m][n] 

# print(findMaxForm(["10", "0001", "111001", "1", "0"],5,3))

# def isMatch(text,pattern):
#     memo=dict()
#     def dp(i,j):
#         if (i,j) in memo:return memo[(i,j)]
#         if j==len(pattern):return i==len(text)
#         first=i<len(text) and pattern[j] in{text[i],'.'}
#         if j<=len(pattern)-2 and pattern[j+1]=='*':
#             ans=dp(i,j+2) or first and dp(i+1,j)
#         else:
#             ans=first and dp(i+1,j+1)
#         memo[(i,j)]=ans
#         return ans
#     return dp(0,0)
# print(isMatch("aaa","a*"))

# class Solution:
#     def isMatch(self, s,p):
#         if not p: return not s
#         if not s and len(p) == 1: return False 
#         nrow = len(s) + 1
#         ncol = len(p) + 1

#         dp = [[False for c in range(ncol)] for r in range(nrow)]
        
#         dp[0][0] = True

#         for c in range(2, ncol):
#             j = c-1
#             if p[j] == '*': dp[0][c] = dp[0][c-2]#主要解决p为,a*，a*a*这种其实可以匹配空
#         for r in range(1, nrow):
#             i = r-1
#             for c in range(1, ncol):
#                 j = c-1
#                 if s[i] == p[j] or p[j] == '.':
#                     dp[r][c] = dp[r-1][c-1]
#                 elif p[j] == '*':
#                     if p[j-1] == s[i] or p[j-1] == '.':
#                         dp[r][c] = dp[r-1][c] or dp[r][c-2]
#                     else:
#                         dp[r][c] = dp[r][c-2]
#                 else:
#                     dp[r][c] = False
#         return dp[nrow-1][ncol-1]
# ss=Solution()
# print(ss.isMatch("aacccc","aac*"))# aac,aac*


# def compute(S,T):
#     windows=dict()
#     needs=dict()
#     left=right=0
#     void=0
#     res=float("inf")
#     start=0
#     for i in T:
#         if i in needs:
#             needs[i]+=1
#         else:
#             needs[i]=1
#     while right<len(S):
#         c=S[right]
#         right+=1
#         if c in needs:
#             if c in windows:
#                 windows[c]+=1
#             else:
#                 windows[c]=1
#             if windows[c]==needs[c]:
#                 void+=1
#             while void==len(needs):
#                 d=S[left]
#                 left+=1
#                 if right-left<res:
#                     res=right-left
#                     start=left
#                 if d in needs:
#                     if windows[d]==needs[d]:
#                         void-=1
#                     windows[d]-=1
#     return S[start-1,start+res] if res!=float("inf") else ""

        
# print(compute("ADOBECODEBANC", "ABC"))

# class Solution:
#     def minimumTotal(self, triangle):
#         n=len(triangle)
#         dp=[[0 for t in range(n)] for i in range(n)]
#         dp[-1]=triangle[-1]
#         for z in range(n-1,-1,-1):
#             t=len(triangle[z])
#             for u in range(t-1):
#                 dp[z-1][u]=min(dp[z][u],dp[z][u+1])+triangle[z-1][u]
#         return dp[0][0]
# ss=Solution()
# print(ss.minimumTotal([[2],[3,4],[6,5,7],[4,1,8,3]]))
# def partition(arr,left,right):
#     povit=arr[right]
#     i=left-1
#     for j in range(left,right):
#         if povit>arr[j]:
#             i+=1
#             arr[i],arr[j]=arr[j],arr[i]
#     arr[i+1],arr[right]=arr[right],arr[i+1]
#     return i+1       
# def quicksort(arr,left,right):
#     if left<right:
#         q=partition(arr,left,right)
#         quicksort(arr,q+1,right)
#         quicksort(arr,left,q-1)
#     return arr
# arr=[6, 12, 27, 34, 21, 4, 9, 8, 11, 54, 3, 7, 39] 
# print(quicksort(arr,0,len(arr)-1))
# class Solution:
#     def sortColors(self, nums)-> None:
#         p0=cur=0
#         p2=len(nums)-1
#         while  cur<=p2:
#             if nums[cur]==0:
#                 nums[cur],nums[p0]=nums[p0],nums[cur]
#                 cur+=1
#                 p0+=1
#             elif nums[cur]==1:
#                 cur+=1
#             else:
#                 nums[cur],nums[p2]=nums[p2],nums[cur]
#                 p2-=1
#         return nums
# ss=Solution()
# print(ss.sortColors([2,0,2,1,1,0]))


# def sortColors(nums):
#     p0=cur=0
#     p2=len(nums)-1
#     while cur<p2:
#         if nums[cur]==0:
#             nums[cur],nums[p0]=nums[p0],nums[cur]
#             cur+=1
#             p0+=1
#         elif nums[cur]==1:
#             cur+=1
#         else:
#             nums[cur],nums[p2]=nums[p2],nums[cur]
#             p2-=1
#     return nums
# print(sortColors([2,0,2,1,1,0]))

# from Tree import TreeNode,TreeNodeTools
# class Solution:
#     def flatten(self, root: TreeNode) -> None:
#         """
#         Do not return anything, modify root in-place instead.
#         """
#         temp=TreeNode()
#         res=temp
#         def dfs(root):
#             if root==None:return None
#             nonlocal temp
#             temp.right=TreeNode(root.val)
#             temp=temp.right
#             dfs(root.left)
#             dfs(root.right)
#         dfs(root)
#         root.left=None
#         root.right=res.right.right

# ss=Solution()
# root=TreeNodeTools().createTreeByrow([1,2,5,3,4,'null',6],0)
# ss.flatten(root)
# TreeNodeTools().printfH(root)
# print(ss.flatten(root))

# def onezerobag(N,W,wt,val):
#     dp=[[0 for i in range(W+1)] for i in range(N+1)]
#     for w in range(W+1):#优先遍历容量
#         for i in range(1,N+1):#再遍历数量
#             if w<wt[i-1]:
#                 dp[i][w]=dp[i-1][w]
#             else:
#                 dp[i][w]=max(dp[i-1][w],dp[i-1][w-wt[i-1]]+val[i-1])
#     return dp[N][W]
# print(onezerobag(3,5,[2,1,3],[4,2,3]))


# from Tree import TreeNode,TreeNodeTools
# class Solution(object):
# 	def buildTree(self, preorder, inorder):
# 		if not (preorder and inorder):
# 			return None
# 		# 根据前序数组的第一个元素，就可以确定根节点	
# 		root = TreeNode(preorder[0])
# 		# 用preorder[0]去中序数组中查找对应的元素
# 		mid_idx = inorder.index(preorder[0])
# 		# 递归的处理前序数组的左边部分和中序数组的左边部分
# 		# 递归处理前序数组右边部分和中序数组右边部分
# 		root.left = self.buildTree(preorder[1:mid_idx+1],inorder[:mid_idx])
# 		root.right = self.buildTree(preorder[mid_idx+1:],inorder[mid_idx+1:])
# 		return root
# ss=Solution()
# print(ss.buildTree([1,2,4,5,3,6,7],[4,2,5,1,6,3,7]))



# class Solution:
#     def isValid(self, s: str) -> bool:
#         dic = {'{': '}',  '[': ']', '(': ')','?':'?'}
#         stack = ['?']
#         for c in s:
#             if c in dic: stack.append(c)
#             elif dic[stack.pop()] != c: return False 
#         return len(stack) == 1
# ss=Solution()
# print(ss.isValid("?"))
# from Tree import TreeNode
# class Solution:
#     def buildTree(self, preorder, inorder) -> TreeNode:
#         if not preorder or not inorder:
#             return None
#         root=TreeNode(preorder[0])
#         mid_idx=inorder.index(preorder[0])
#         root.left=self.buildTree(preorder[1:mid_idx+1],inorder[:mid_idx])
#         root.right=self.buildTree(preorder[mid_idx+1:],inorder[mid_idx+1:])
#         return root
# ss=Solution()
# print(ss.buildTree([3,9,20,15,7],[9,3,15,20,7]))

# from Tree import TreeNode,TreeNodeTools

# class Solution:
#     def isSymmetric(self, root: TreeNode) -> bool:
#         if not root:return True
#         def dfs(right,left):
#             if not right and not left:
#                 return True
#             if not right or not left:
#                 return False
#             if right.val!=left.val:
#                 return False
#             return dfs(left.right,right.left) and dfs(left.left,right.right)
#         return dfs(root.right,root.left)

# tools=TreeNodeTools()
# root=tools.createTreeByrow("[1,2,2,2,null,2,null,null,null,null,null]")#这个就是典型的问题，后面null没有补全
# ss=Solution()
# print(ss.isSymmetric(root))

# from Tree import TreeNode,TreeNodeTools
# class Solution:
#     def isValidBST(self, root: TreeNode) -> bool:
#         pre=-float("inf")
#         def dfs(root):
#             nonlocal pre
#             if not root:return True
#             if not dfs(root.left):
#                 return False
#             if root.val<=pre:
#                 return False
#             pre=root.val
#             return dfs(root.right)
#         return dfs(root)
# tools=TreeNodeTools()
# root=tools.createTreeByrow("[1,2,2,2,null,2,null,null,null,null,null]")#这个就是典型的问题，后面null没有补全
# ss=Solution()
# print(ss.isValidBST(root))

# def twofind(llist,target):
#     left=0
#     right=len(llist)-1
#     while left<=right:
#         mid=left+(right-left)//2
#         if llist[mid]<target:
#             left=mid+1
#         elif llist[mid]>target:
#             right=mid-1
#         elif llist[mid]==target:
#             res=[mid,mid]
#             temp=mid-1
#             while temp>=0 and llist[temp]==target:
#                 res[0]=temp
#                 temp-=1
#             temp=mid+1
#             while temp<len(llist) and llist[temp]==target:
#                 res[1]=temp
#                 temp+=1
#             return res
#     return [-1,-1]
# print(twofind([1],1))
# class Solution:
#     def search(self, nums, target: int) -> int:
#         left=0
#         right=len(nums)-1

#         while left<=right:
#             mid=left+(right-left)//2
#             #判断左边一半是有序的
#             if nums[mid]==target:
#                 return mid
#             if nums[mid]>=nums[0]:
#                 if nums[0]<=target<nums[mid]:
#                     right=mid-1
#                 else:
#                     left=mid+1
#             #右边部分是有序的
#             else:
#                 if nums[mid]<target<=nums[len(nums)-1]:
#                     left=mid+1
#                 else:
#                     right=mid-1
#         return -1
# ss=Solution()
# print(ss.search([3,1],1))

# class Solution:
#     def findWords(self, board, words):
#         #首先建立字典树
#         tree={}
#         trie=tree
#         for word in words:
#             tree=trie
#             for i in word:
#                 if i not in tree:
#                     tree[i]={}
#                 tree=tree[i]
#             tree['#']=word
#         res=[]
#         def dfs(t,u,tr):
#             if not 0<=t<len(board) or not 0<=u<len(board[0]) or board[t][u] not in tr:return
#             if '#' in tr[board[t][u]]:
#                 res.append(tr[board[t][u]]['#'])
#                 tr[board[t][u]].pop('#')
#             board[t][u],temp='/',board[t][u]
#             dfs(t+1,u,tr[temp]) or dfs(t-1,u,tr[temp]) or dfs(t,u+1,tr[temp]) or dfs(t,u-1,tr[temp])
#             board[t][u]=temp

#         #采用回溯法去搜索
#         for t in range(len(board)):
#             for u in range(len(board[0])):
#                 if board[t][u] in trie:
#                     dfs(t,u,trie)
#         return res
# ss=Solution()
# print(ss.findWords([["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]],["oath","pea","eat","rain","oat"]))

# from Tree import TreeNode,TreeNodeTools

# class Solution:
#     def __init__(self):
#         self.ans=0
#     def helper(self,node,curr):
#         if not node:
#             return 
#         if node.val==curr:
#             self.ans+=1

#         self.helper(node.left,curr-node.val)
#         self.helper(node.right,curr-node.val)

#     def pathSum(self, root: TreeNode, sum: int) -> int:
#         if not root:
#             return 0
#         self.helper(root,sum)
#         self.pathSum(root.left,sum)
#         self.pathSum(root.right,sum)

#         return self.ans


# ss=Solution()
# root=TreeNodeTools().createTreeByrow('[1,-2,3,1,3,-2,null,-1,null,null,null,null,null,null,null]')
# print(ss.pathSum(root,-1))

#'[10,5,-3,3,2,null,11,3,-2,null,1,null,null,null,null,null,null,null,null]' 8

# from List import ListNode,ListNodeTools

# def detectCycle(head):
#     fast, slow = head, head
#     while True:
#         if not (fast and fast.next): return None
#         fast, slow = fast.next.next, slow.next
#         if fast == slow: break
#     fast = head
#     while fast != slow:
#         fast, slow = fast.next, slow.next
#     return fast

# print(detectCycle(ListNodeTools().list9_h()))
# from Tree import TreeNode,TreeNodeTools
# class Solution:
#     def maxPathSum(self, root: TreeNode) -> int:
#         self.maxx=root.val
#         def treehigh(root):
#             if not root:return 0
#             left=max(treehigh(root.left),0)
#             right=max(treehigh(root.right),0)

#             self.maxx=max(self.maxx,right+left+root.val)
#             return root.val+max(left,right)
#         treehigh(root)
#         return self.maxx
# ss=Solution()
# t=TreeNodeTools()
# root=t.createTreeByrow('[1,2,null,3,null,4,null,5,null,null,null]')
# print(ss.maxPathSum(root))
# from Tree import TreeNode,TreeNodeTools
# class Solution:
#     def buildTree(self, preorder,inorder) -> TreeNode:
#         if not preorder or not inorder:return None
#         root=TreeNode(preorder[0])
#         index=inorder.index(preorder[0])
#         root.left=self.buildTree(preorder[1:index+1],inorder[:index])
#         root.right=self.buildTree(preorder[index+1:],inorder[index+1:])
#         return root
# root=Solution().buildTree([3,9,20,15,7],[9,3,15,20,7])
# TreeNodeTools().printfH(root)
# import collections
# class Solution:
#     def canFinish(self, numCourses: int, prerequisites) -> bool:
#         edges = collections.defaultdict(list)
#         indeg = [0] * numCourses
#         for c in prerequisites:
#             edges[c[1]].append(c[0])
#             indeg[c[0]] += 1
#         q = collections.deque([u for u in range(numCourses) if indeg[u] == 0])
#         visited = 0
#         while q:
#             visited+=1
#             c= q.popleft()
#             for i in edges[c]:
#                 indeg[i]-=1
#                 if indeg[i]==0:
#                     q.append(i)
#         return visited==numCourses
# print(Solution().canFinish(4,[[1,0],[2,0],[3,1],[3,2]]))

# class Solution:
#     def decodeString(self, s: str) -> str:
#         stack,multi,res=[],0,''
#         for i in s:
#             if i=='[':
#                 stack.append([multi,res])
#                 multi,res=0,''
#             elif i==']':
#                 pre_multi,pre_res = stack.pop()
#                 res = pre_res+pre_multi*res
#             elif '0'<=i<='9':
#                 multi= 10*multi+int(i)
#             else:
#                 res+=i
#         return res

# ss=Solution()
# ss.decodeString("3[a2[c]]")

# class Solution:
#     def moveZeroes(self, nums) -> None:
#         if not nums:
#             return 0
#         j = 0
#         for i in range(len(nums)):
#             if nums[i]:
#                 nums[j],nums[i] = nums[i],nums[j]
#                 j += 1
# ss=Solution()
# print(ss.moveZeroes([1,1,0,3,12]))
# from List import ListNode,ListNodeTools
# class Solution:
#     def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
#         pp=ListNode(-1)
#         re=pp
#         while l1 and l2:
#             if l1.val>l2.val:
#                 pp.next=l2
#                 l2=l2.next
#             else:
#                 pp.next=l1
#                 l1=l1.next
#             pp=pp.next
#         if l1:
#             pp.next=l1
#         else:
#             pp.next=l2
#         return re.next
#     def mergeKLists(self, lists) -> ListNode:
#         amount=len(lists)
#         interval=1
#         while interval<amount:
#             for i in range(0,amount-interval,interval*2):
#                 lists[i]=self.mergeTwoLists(lists[i],lists[i+interval])
#             interval*=2
#         return lists[0] if amount>0 else None
# ss=ListNodeTools()
# l1=ss.create([1,4,5])
# l2=ss.create([1,3,4])
# l3=ss.create([2,6])
# l4=ss.create([1,2,6])
# l5=ss.create([2,6,7])
# ss.printf(Solution().mergeKLists([l1,l2,l3,l4,l5]))
# class Solution:
#     def subarraySum(self, nums, k: int) -> int:
#         left=0
#         right=1
#         res=0
#         while left<right and right<len(nums)+1:
#             su=0
#             for i in range(left,right):
#                 su+=nums[i]
#             if su==k:
#                 res+=1
#                 left+=1
#             elif su>k:
#                 left+=1
#             else:
#                 right+=1
#         return res
# ss=Solution()
# print(ss.subarraySum([-1,-1,1],0))
# class Solution:
#     def minMeetingRooms(self, intervals) -> int:
#         events = [(i[0],1) for i in intervals] + [(i[1],-1) for i in intervals]
#         events.sort()
#         res,cur = 0,0
#         for _,e in events:
#             cur += e
#             res = max(res,cur)
#         return res
# ss=Solution()
# print(ss.minMeetingRooms([[0,30],[5,10],[15,20]]))
# from List import ListNode,ListNodeTools
# class Solution:
#     def isPalindrome(self, head: ListNode) -> bool:
#         if not head or not head.next:
#             return True
#         slow=head
#         fast=head
#         pre=head
#         prepre=None
#         while fast and fast.next:
#             pre=slow
#             slow=slow.next
#             fast=fast.next.next
#             pre.next=prepre
#             prepre=pre
#         if fast:
#             slow=slow.next
#         while pre and slow:
#             if pre.val!=slow.val:
#                 return False
#             pre=pre.next
#             slow=slow.next
#         return True

# ss=ListNodeTools()
# root3=ss.create([1,2,2,1])
# tools=Solution()
# print(tools.isPalindrome(root3))

# class Solution:
#     def removeInvalidParentheses(self, s:str):
#         def isValid(s:str)->bool:
#             cnt = 0
#             for c in s:
#                 if c == "(": cnt += 1
#                 elif c == ")": cnt -= 1
#                 if cnt < 0: return False  # 只用中途cnt出现了负值，你就要终止循环，已经出现非法字符了
#             return cnt == 0

#         # BFS
#         level = {s}  # 用set避免重复
#         while True:
#             valid = list(filter(isValid, level))  # 所有合法字符都筛选出来
#             if valid: return valid # 如果当前valid是非空的，说明已经有合法的产生了
#             # 下一层level
#             next_level = set()
#             for item in level:
#                 for i in range(len(item)):
#                     if item[i] in "()":                     # 如果item[i]这个char是个括号就删了，如果不是括号就留着
#                         next_level.add(item[:i]+item[i+1:])
#             level = next_level

# ss=Solution()
# print(ss.removeInvalidParentheses("()())()"))

# class Solution:
#     def maximalRectangle(self, matrix) -> int:
#         if not matrix:return 0
#         m,n=len(matrix),len(matrix[0])
#         # 记录当前位置上方连续“1”的个数
#         pre=[0]*(n+1)
#         res=0
#         for i in range(m):
#             for j in range(n):
#                 # 前缀和
#                 pre[j]=pre[j]+1 if matrix[i][j]=="1" else 0

#             # 单调栈
#             stack=[-1]
#             for k,num in enumerate(pre):
#                 while stack and pre[stack[-1]]>num:
#                     index=stack.pop()
#                     res=max(res,pre[index]*(k-stack[-1]-1))
#                 stack.append(k)
#         return res
# ss=Solution()
# matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
# print(ss.maximalRectangle(matrix))

# class Solution:
#     def maximalRectangle(self, matrix) -> int:
#         if not matrix:return 0
#         m,n=len(matrix),len(matrix[0])
#         # 记录当前位置上方连续“1”的个数
#         pre=[0]*(n+1)
#         res=0
#         for i in range(m):
#             for j in range(n):
#                 # 前缀和
#                 pre[j]=pre[j]+1 if matrix[i][j]=="1" else 0
#             # 单调栈
#             stack=[-1]
#             for k,num in enumerate(pre):
#                 while stack and pre[stack[-1]]>num:
#                     index=stack.pop()
#                     res=max(res,pre[index]*(k-stack[-1]-1))
#                 stack.append(k)

#         return res
# ss=Solution()
# print(ss.maximalRectangle([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]))

# class Solution:
#     def largestRectangleArea(self, heights) -> int:
#         stack=[]
#         res=0
#         heights=[0]+heights+[0]
#         for i in range(len(heights)):
#             while stack and heights[stack[-1]]>heights[i]:
#                 temp=stack.pop()
#                 res=max(res,(i-stack[-1]-1)*heights[temp])

#             stack.append(i)
#         return res
# ss=Solution()
# print(ss.largestRectangleArea([2,1,5,6,2,3]))

# def dailyTemperatures(temperatures):
#     stack=[]
#     res=[0 for i in range(len(temperatures))]
#     for i,v in enumerate(temperatures):
#         while stack and v>temperatures[stack[-1]]:
#             node=stack.pop()
#             res[node]=i-node
#         stack.append(i)
#     return res 
# print(dailyTemperatures([73,74, 75, 71, 69, 72, 76, 73]))
 
