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
    ListNode* removeNthFromEnd(ListNode* head, int n) {
    //     ListNode* prev = nullptr;
    //     ListNode* temp = head;
    //     while(temp){
    //         ListNode* front = temp->next;
    //         temp->next = prev;
    //         prev = temp;
    //         temp = front;
    //     }
    //     ListNode* dummy = new ListNode(0);
    //     dummy->next = prev;
    //     ListNode* current = dummy;   // ✅ start from dummy

    //     int count = 0;
    //     while (current->next) {
    //         if (count == n - 1) {
    //         ListNode* p = current->next;
    //         current->next = current->next->next;
    //         delete p;
    //         break;
    //         }
    //         count++;
    //         current = current->next;
    //     }
    //     ListNode* newprev = nullptr;
    //     ListNode* newptr=dummy->next;
    //     while(newptr){
    //         ListNode* newfront = newptr->next;
    //         newptr->next = newprev;
    //         newprev = newptr;
    //         newptr = newfront;
        
    //     }
    //     return newprev;

    // }
    ListNode* dummy = new ListNode(0);
    dummy->next = head;


    ListNode* front = dummy;
    ListNode* back = dummy;
    for(int i = 0; i<=n; i++){
        front = front->next;
    }
    while(front!= nullptr){
        front = front->next;
        back = back->next;
    }
    ListNode* nodetodelete = back->next;
    back->next = back->next->next;

    delete nodetodelete;
    ListNode* newhead = dummy->next;
    delete dummy;
    return newhead;
    }
};
