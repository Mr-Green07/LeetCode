/**
 * Definition for singly-linked list.
 * struct ListNode {
 * int val;
 * ListNode *next;
 * ListNode() : val(0), next(nullptr) {}
 * ListNode(int x) : val(x), next(nullptr) {}
 * ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
#include <stack>

class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        std::stack<int> s1, s2;

        // 1. Push all values onto stacks
        while (l1) {
            s1.push(l1->val);
            l1 = l1->next;
        }
        while (l2) {
            s2.push(l2->val);
            l2 = l2->next;
        }

        int sum = 0;
        int carry = 0;
        ListNode* head = nullptr;

        // 2. Process stacks
        while (!s1.empty() || !s2.empty() || carry) {
            int val1 = 0;
            if (!s1.empty()) {
                val1 = s1.top();
                s1.pop();
            }

            int val2 = 0;
            if (!s2.empty()) {
                val2 = s2.top();
                s2.pop();
            }

            sum = val1 + val2 + carry;
            carry = sum / 10;
            
            // 3. Create node and insert at head (Head Insert)
            ListNode* newNode = new ListNode(sum % 10);
            newNode->next = head;
            head = newNode;
        }

        return head;
    }
};
