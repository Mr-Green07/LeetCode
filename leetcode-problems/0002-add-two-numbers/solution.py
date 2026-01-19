# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for singly-linked list.
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)  # Placeholder head
        current = dummy
        carry = 0
        
        # Run loop while l1 or l2 exists, or we have a carry left over
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            
            # Calculate sum and new carry
            total = val1 + val2 + carry
            carry = total // 10
            new_digit = total % 10
            
            # Add new node to result list
            current.next = ListNode(new_digit)
            current = current.next
            
            # Move pointers forward if available
            if l1: l1 = l1.next
            if l2: l2 = l2.next
            
        return dummy.next
