class Solution {
public:
    bool canJump(vector<int>& nums) {
        int target = nums.size() -1;
        int maxR = 0;
        for(int i = 0; i<= maxR; i++){
            maxR = max(maxR, i + nums[i]);
            if(maxR >= target){
                return true;
            }
        }
        return false;
    }
};
