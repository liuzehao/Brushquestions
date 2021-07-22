class TreeNode:
    def __init__(self,x=0):
        self.val=x
        self.left=None
        self.right=None
    #打印树
class TreeNodeTools:
    def printfDfs(self,root):
        if not root: return
        #前序遍历的代码
        print(root.val)
        self.printfDfs(root.right)
        self.printfDfs(root.left)
    #广度优先遍历
    def printfH(self,root):
        if not root: return "[]"
        queue = []
        queue.append(root)
        res = []
        while queue:
            node = queue.pop(0)
            if node:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else: res.append("null")
        return '[' + ','.join(res) + ']'
    #层序遍历
    def printLevel(self,root):
        quene=[]
        quene.append(root)
        res=[]
        while len(quene)>0:
            temp=[]
            for i in range(len(quene)):
                node=quene.pop(0)
                temp.append(node.val)
                if node.left:
                    quene.append(node.left)
                if node.right:
                    quene.append(node.right)
            res.append(temp[:])
        return res
    #行序遍历建树
    def createTreeByrow(self,data):
        vals, i = data[1:-1].split(','), 1
        root = TreeNode(int(vals[0]))
        queue = []
        queue.append(root)
        while queue:
            node = queue.pop(0)
            if vals[i] != "null":
                node.left = TreeNode(int(vals[i]))
                queue.append(node.left)
            i += 1
            if vals[i] != "null":
                node.right = TreeNode(int(vals[i]))
                queue.append(node.right)
            i += 1
        return root
    #求树高
    def treeHigh(self,root):
        if not root:return 0
        left=self.treeHigh(root.right)
        right=self.treeHigh(root.left)
        return max(left,right)+1
    
if __name__ == "__main__":
    ss=TreeNodeTools()
    root3=ss.createTreeByrow("[1,2,3,null,null,4,5,null,null,null,null]")
    # print(ss.printfH(root3))
    # print(ss.printLevel(root3))
    ss.printfDfs(root3)
    # print(ss.treeHigh(root3))
'''
构造树的形式
        1
    2        3
   n  n    4    5
          n n  n n
dfs:
前序：1 2 3 4 5
中序：2 1 4 5 3
后序：2 4 5 3 1

bfs:
1 2 3 4 5
'''