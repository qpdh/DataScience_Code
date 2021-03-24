def pdf(x: float) -> float:
    return 1 if 2.0 <= x <= 5.0 else 0


# fill in with your code here
def cdf(x: float) -> float:
    if x < 2.0:
        return 0
    elif 2.0 <= x <= 5.0:
        return (x - 2.0) / 3
    else:
        return 1


# fill in with your code here
# print
print("pdf(2.5)=", pdf(2.5))
print("cdf(2.5)=", cdf(2.5))
