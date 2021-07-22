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

