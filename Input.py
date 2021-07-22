#输入输出
import sys
#载入调试文件
# sys.stdin = open('input.txt', 'r')
# #多个数字
# b, a = map(int, sys.stdin.readline().strip().split(' '))
# print(b,a)
# #单个数字
# a=int(sys.stdin.readline().strip())
# print(a)
# #多个数字
# tmp = list(map(int, sys.stdin.readline().strip().split(' ')))
# print(tmp)
# #矩阵n行
# n=4
# llist=[]
# for i in range(n):
#     tmp = list(map(int, sys.stdin.readline().strip().split(' ')))
#     llist.append(tmp)
# print(llist)
class Input:
    def __init__(self):
        sys.stdin = open('input.txt', 'r')
    def read_two(self):
        b, a = map(int, sys.stdin.readline().strip().split(' '))
        return (b,a)
    def read_one(self):
        a=int(sys.stdin.readline().strip())
        return a
    def read_list(self):
        tmp = list(map(int, sys.stdin.readline().strip().split(' ')))
        return tmp
    def read_nlist(self,n):
        llist=[]
        for i in range(n):
            tmp = list(map(int, sys.stdin.readline().strip().split(' ')))
            llist.append(tmp)
        return llist
if __name__ == "__main__":
    ss=Input()
    llist=ss.read_nlist(4)
    print(llist)