/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        // Dummy node simplifies edge cases at the head
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* groupPrev = dummy;

        while (true) {
            // Step 1: Check if k nodes exist ahead
            ListNode* kth = getKth(groupPrev, k);
            if (!kth) break;  // Fewer than k nodes remain → stop

            ListNode* groupNext = kth->next;  // Node after the group

            // Step 2: Reverse the k nodes
            ListNode* prev = groupNext;  // Connect tail to next group
            ListNode* curr = groupPrev->next;

            while (curr != groupNext) {
                ListNode* tmp = curr->next;
                curr->next = prev;
                prev = curr;
                curr = tmp;
            }

            // Step 3: Re-connect groupPrev to the new head of reversed group
            ListNode* newHead = kth;               // kth becomes new group head
            ListNode* newTail = groupPrev->next;   // original head becomes new tail
            groupPrev->next = newHead;
            groupPrev = newTail;                   // advance groupPrev to end of reversed group
        }

        return dummy->next;
    }

private:
    // Returns the kth node from 'curr', or nullptr if fewer than k nodes
    ListNode* getKth(ListNode* curr, int k) {
        while (curr && k > 0) {
            curr = curr->next;
            k--;
        }
        return curr;
    }
};
