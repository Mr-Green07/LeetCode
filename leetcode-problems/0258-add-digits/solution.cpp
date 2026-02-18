class Solution {
public:
    int add(int num){
        int sum = 0;
        if (num<=0){
            return 0;
        }
       int d = num %10;
        sum+=d;
        return sum + add(num/10);
    }
    int addDigits(int num) {
        int sum = 0;
        if(num<10){
            return num;
        }
        // while(num){
        //     int n = num % 10;
        //     sum+=n;
        //     num =num/10;
        // }
        int a = add(num);
        return addDigits(a);
    }
};
