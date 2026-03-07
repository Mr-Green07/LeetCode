#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    bool canJump(vector<int>& nums) {
        int max_reach = 0;
        int n = nums.size();
        
        for (int i = 0; i < n; i++) {
            // If the current index is beyond our furthest reach, we are stuck
            if (i > max_reach) {
                return false;
            }
            
            // Update the furthest we can jump from here
            max_reach = max(max_reach, i + nums[i]);
            
            // Optimization: If we can already reach the end, break early
            if (max_reach >= n - 1) {
                return true;
            }
        }
        
        return true;
    }
};

