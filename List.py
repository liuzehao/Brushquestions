class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class ListNodeTools:
    def list9_h(self):
        llist9=ListNode(9)
        llist10=ListNode(10)
        llist11=ListNode(11)
        llist12=ListNode(12)
        llist13=ListNode(13)
        llist14=ListNode(14)
        llist15=ListNode(15)
        llist9.next=llist10
        llist10.next=llist11
        llist11.next=llist12
        llist12.next=llist13
        llist13.next=llist14
        llist14.next=llist15
        llist15.next=llist11
        return llist9
    def list15(self):
        llist1=ListNode(1)
        llist2=ListNode(2)
        llist3=ListNode(3)
        llist4=ListNode(4)
        llist5=ListNode(5)
        llist1.next=llist2
        llist2.next=llist3
        llist3.next=llist4
        llist4.next=llist5
        return llist1
    def printf(self,cur):
        while cur:
            print(cur.val)
            cur=cur.next
    def create(self,llist):
        first=ListNode(-1)
        temp_f=first
        for i in range(len(llist)):
            temp_l=ListNode(llist[i])
            temp_f.next=temp_l
            temp_f=temp_f.next
        return first.next

if __name__ == "__main__":
    ss=ListNodeTools()
    root3=ss.create([1,2,3,4,5,6])
    ss.printf(root3)