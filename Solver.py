# Function checker => checks is last element breaks the pattern or not
def checker(per):
    i=len(per)-1
    for j in range(i):
        if( i-j == abs(per[i]-per[j])):
            return False
    return True
# Function solver => Generates the solution list.
# n => dimension of the chessboard
# ans => flag to determine whether solution is possible or not
def solver(per, n, ans=False):
    if(len(per)==n):
        return per
    for i in range(n):
        if i not in per:
            per.append(i)
            if(checker(per)):
                ans=solver(per,n)
                if(ans):
                    return ans
            per.pop()
    return ans
if __name__ == '__main__':
    print("Enter n: ", end='')
    k=int(input())
    print(solver([],k))
