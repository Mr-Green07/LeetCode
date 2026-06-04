/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
*/

class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head) return nullptr;
        
        // Step 1: Map each original node to its copy
        unordered_map<Node*, Node*> oldToNew;
        Node* curr = head;
        while (curr) {
            oldToNew[curr] = new Node(curr->val);
            curr = curr->next;
        }
        
        // Step 2: Assign next and random pointers
        curr = head;
        while (curr) {
            if (curr->next)
                oldToNew[curr]->next = oldToNew[curr->next];
            if (curr->random)
                oldToNew[curr]->random = oldToNew[curr->random];
            curr = curr->next;
        }
        
        return oldToNew[head];
    }
};
