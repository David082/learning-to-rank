package leetcode.cn.medium;

/*
 * Created by yu_wei on 2019/3/14.
 */
class ListNode {
    int val;
    ListNode next;

    public ListNode(int x) {
        this.val = x;
    }
}

public class Medium {

    public static void main(String[] args) {
        ListNode first = new ListNode(1);
        first.next = new ListNode(2);
        first.next.next = new ListNode(3);
        first.next.next.next = new ListNode(4);

        showLink(first);
        System.out.println();
        showLink(removeNthFromEnd(first, 2));
    }

    /**
     * Num019
     * 给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。
     *
     * @param head
     * @param n
     * @return
     */
    public static ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        int length = 0;
        ListNode first = head;
        while (first != null) {
            length++;
            first = first.next;
        }

        length -= n;
        first = dummy;
        while (length > 0) {
            length--;
            first = first.next;
        }
        first.next = first.next.next;

        return dummy.next;
    }

    /**
     * 打印链表
     *
     * @param first
     */
    public static void showLink(ListNode first) {
        ListNode node = first;
        while (node != null) {
            System.out.print(node.val + "\t");
            node = node.next;
        }
    }
}
