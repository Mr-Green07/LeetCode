class Solution {
public:
    int maxProduct(vector<int>& nums) {
        ranges::sort(nums);
        int n = nums.size();
        return ((nums[n - 1] - 1) *( nums[n - 2]-1));
        
    }
};
