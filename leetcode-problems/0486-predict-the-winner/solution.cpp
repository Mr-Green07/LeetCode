class Solution {
    int memo[21][21];
public:
    bool predictTheWinner(vector<int>& nums) {
        int n = nums.size();
        memset(memo, -1, sizeof(memo));
        return maxDiff(nums, 0, n - 1) >= 0;
        
    }
    int maxDiff(vector<int>& nums, int i, int j){
        if(i == j) return nums[i];
        if(memo[i][j] != -1) return memo[i][j];
        int pickleft = nums[i]  - maxDiff(nums, i + 1, j);
        int pickright = nums[j] - maxDiff(nums, i, j - 1);
        return memo[i][j] = max(pickleft, pickright);
    }
};
