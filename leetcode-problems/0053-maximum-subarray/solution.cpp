class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        if(n == 1) return nums[0];
        vector<int> dp(n+1, 0);
        dp[0] = nums[0];
        int maxsum = dp[0];

        for(int i = 1; i< n; ++i){
            dp[i] = max(dp[i - 1] + nums[i], nums[i]);
            maxsum = max(maxsum, dp[i]);
        }
        return maxsum;
    }
};
