class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int n = nums.size();
        int sum = 0;
        for(int num: nums){
            sum += num;
        }
        if(sum % 2 != 0){
            return false;
        }
        int target = sum /2;
        vector<bool> t(target + 1, false);
        t[0]= true;

        for(int i = 0; i<n; i++){
            for(int j = target; j>=nums[i]; j--){
                t[j] = t[j] || t[j-nums[i]];

            }
        }
        return t[target];

    }
};
