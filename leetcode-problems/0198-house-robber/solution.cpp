class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        if(nums.empty()) return 0;

        // int prev_max = 0;
        // int curr_max = 0;
        // for(int i: nums){
        //     int new_max = max(curr_max, prev_max + i);

        //     prev_max = curr_max;
        //     curr_max = new_max;
        // }
        // return curr_max;
        vector<int> dp(n + 1);
        dp[0]=0;
        dp[1]= nums[0];
        for(int i = 2; i<=n; i++){
            dp[i] = max(dp[i -2] + nums[i-1], dp[i - 1]);
        }
        return dp[n];
        
    }
};
