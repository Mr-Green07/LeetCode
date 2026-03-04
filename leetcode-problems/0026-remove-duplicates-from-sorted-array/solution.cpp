#include <vector>

using namespace std;

class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if (nums.empty()) return 0;
        
        int insertIndex = 1; // The Bouncer
        
        // The Scout loops from 1 to the end
        for (int i = 1; i < nums.size(); i++) {
            // Found a new unique element
            if (nums[i] != nums[i - 1]) {
                nums[insertIndex] = nums[i]; // Move it to the front
                insertIndex++;               // Advance the insertion point
            }
        }
        
        return insertIndex;
    }
};

