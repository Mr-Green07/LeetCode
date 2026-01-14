class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        map<long long, int> preSum;
        long long sum = 0;
        int  maxlen = 0;
        preSum[0] =1;
        for(int i = 0; i < nums.size(); i++){
            sum += nums[i];
            if(preSum.find(sum-k) != preSum.end()){
                maxlen += preSum[sum-k];
            }
            preSum[sum]++;

        }
        return maxlen;
        
    }
};
