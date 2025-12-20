class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        
        int result ;
            for(int j = 0; j <nums.size(); j++){
                if (nums[j]<=0 || nums[j]>nums.size())
                {
                    nums[j] = nums.size()+1;
                    
                }
            }
            for(int i =0;i<nums.size();i++)
            {
                int index = abs(nums[i]);
                if(index >=1 && index<=nums.size())
                {
                    int val = index -1;
                    if(nums[val]>0)
                    {
                        nums[val] = -nums[val];
                    }
                }
            }
            for(int k =0;k<nums.size();k++)
            {
                if(nums[k]>0)
                {
                    return k+1;
                }
            }
            return nums.size()+1;
    }
};
