class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int n = nums.size();

        int sum = 0;
        for(int i = 0; i<n; i++){
            sum += nums[i];
        }
        
        if(sum%2 != 0){
            return false;
        }
        int target = sum /2;
        vector<bool> dp(target+1, false);
        dp[0] = true;
        // for(int i = 0; i<=n; i++){
        //     dp[i][0] = true;
        // }
        for(int i: nums){
            for(int j = target; j>=i; j--){
            //     dp[i][j]= dp[i-1][j];

            //     if(j >= nums[i-1]){

            //         dp[i][j] = dp[i][j] || dp[i - 1][j-nums[i-1]];
            //     } 

                
            // }
                dp[j] = dp[j] || dp[j - i];
            }
        }
        return dp[target];
    }
};
