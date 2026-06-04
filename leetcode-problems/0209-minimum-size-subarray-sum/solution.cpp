class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int left = 0;
        int n = nums.size();
        int min_length = INT_MAX;
        int curr_sum = 0;
       
        for(int i = 0; i< n; i++){
            curr_sum += nums[i];

            while(curr_sum >= target){
                min_length = min(min_length, i-left+1);
                curr_sum -= nums[left];
                left++;
            }
          
        }
        return min_length == INT_MAX ? 0: min_length;
    }
};
