using Plots
gr()
using Statistics
using LinearAlgebra
using ForwardDiff
using LaTeXStrings

#Save most recent figure as a .png file to figures\\filename
function pngsave(filename::String)
    png("statfigures\\"*filename)
end


## Question 1 - Greenwood model

#Plot infected number of susceptibles over time under the Greenwood model, using recursive relation
function Q1(x0::Int,α::Float64;T=5::Int)
    xVals = Float64[x0]
    for t = 1:T
        push!(xVals,α*xVals[t])
    end

    scatter(0:T,xVals,label=:none)
    #title!("Greenwood Model")
    xlabel!("Time")
    ylabel!("Expected Number of Susceptible Individuals")
end

## Question 2 - Reed-Frost model

#Plot expected number of susceptibles and infectives over time under the read-frost model using the recursive relation on the expectation
function Q2()
    α = 0.98
    x0 = 99.0
    y0 = 1.0
    T = 15
    xVals = [x0]
    yVals = [y0]
    for t = 1:T
        push!(xVals,α^yVals[t]*xVals[t])
        push!(yVals,(1-α^yVals[t])*xVals[t])
    end

    plot(0:T,[xVals,yVals],layout=(1,2),title=["Susceptible" "Infected"],xlabel = ["Time" "Time"],label=:none)
end

#plot joint trajectory of expectation  of susceptibles and infectives
function Q2other()
    α = 0.98
    x0 = 99.0
    y0 = 1.0
    T = 15
    next(x,y) = [x*α^y,x*(1-α^y)]# update formula
    traj = [[] for _ in 1:T]
    traj[1] = [x0,y0]
    for t in 2:T
        traj[t] = next(traj[t-1]...)
    end
    plot(first.(traj),last.(traj),label="Model Trajectory")
    scatter!([traj[1][1]],[traj[1][2]],c=:black,label="Starting Point")
    xlabel!("Number of Susceptible")
    ylabel!("Number of Infected")
end
## Question 3

#Generate heatmap of transition probability matrix under Greenwood model
function Q3a(x0,α)
    P = zeros(x0+1,x0+1)

    for i in 1:x0+1
        P[i,i] = α^(i-1)
        for j in 1:i-1
            P[i,j] = binomial(i-1,j-1)*(1-α)^(i-1)*α^(j-1)
        end
    end
    heatmap(P)
    #title!("Heatmap of P, x_0=$x0,α=$α")
    ylabel!("Number of Susceptibles 'from'")
    xlabel!("Number of Susceptibles 'to'")
end

# Use numerical solution to analytical matrix method to evaluate expected number of susceptibles over time under Greenwood model
function Q3c(x0,α;T=5::Int)
    P = zeros(x0+1,x0+1)

    for i in 1:x0+1
        P[i,i] = α^(i-1)
        for j in 1:i-1
            P[i,j] = binomial(i-1,j-1)*(1-α)^(i-j)*α^(j-1)
        end
    end
    e = zeros(1,x0+1)
    e[x0+1] = 1

    v = [x for x in 0:x0]

    vals = [(e*(P^t)*v)[1] for t = 0:T]
    plot(Q1(x0,α,T=T),label="Calculated Recursively")
    plot!(0:T,vals,label="Calculated using Matrices")
    #title!("Greenwood Model, x_0=$x0, alpha=$α")
end

# Use numerical solution to analytical matrix method to evaluate expected number of infectives over time under Greenwood model
function Q3d(x0::Int,xt::Int,α::Float64;T=5::Int,y0=1.0::Float64)
    P = zeros((x0+1)^2,(x0+1)^2)
    States = Dict(i => ((i-1)÷(x0+1),(i-1)%(x0+1)) for i in 1:(x0+1)^2) #get table of conversions between matrix index and state
    for i in 1:(x0+1)^2
        for j in 1:(x0+1)^2
            xnow,ynow = States[i]
            xnext,ynext = States[j]
            if xnext + ynext == xnow
                P[i,j] = binomial(xnow,xnext)*α^xnext*(1-α)^ynext
            end
        end
    end
    Indices = Dict(value => key for (key,value) in States) #give state to return index
    e = zeros(1,(x0+1)^2)
    e[Indices[x0,y0]] = 1
    v = [(i-1)%(x0+1) for i in 1:(x0+1)^2]
    vals = [(e*(P^t)*v)[1] for t = 0:T]
    plot(0:T,[α^(t-1)*(1-α)*xt for t in 0:T],label="Calculated Recursively")
    plot!(0:T,vals,label="Calculated using Matrices")
    ylabel!("Expected Number of Infectives")
    xlabel!("Time")
    #title!("Greenwood Model, x_0=$x0, alpha=$α")
end

## Question 4 - Numerics of Greenwood model
# defined in
p1(i,j,α) = binomial(i,j)*(1-α)^(i-j)*α^j #p_{ij} in EM-4

_p2 = Dict()
function p2(j,t,x0,α) #equation 4.1.6 - p_j^t in EM-4
    #println(j,t)
    if t == 0
        return Float64(j==x0)
    end
    if (j,t,x0,α) in keys(_p2)
        return _p2[(j,t)]
    end
    vals=[0.0]
    #[println(vals,p2(i,t-1)*p1(i,j) for i in (j+1):(x0-(t-1)))]
    [push!(vals,p2(i,t-1,x0,α)*p1(i,j,α)) for i in (j+1):(x0-(t-1))]
    res = sum(vals)
    #res = sum([p2(i,t-1)*p1(i,j) for i in (j+1):(x0-(t-1))])
    _p2[(j,t,x0,α)] = res
    return res
end

Γ(k,n,x0,α) = p2(x0-k,n-1,x0,α)*α^(x0-k) #Γ given in problem statement

#Use Γ function to determine probability of more than 4 new infections
function Q4b(x0,α)
    PWG4 = sum([Γ(k,n,x0,α) for k in 5:x0 for n in 1:x0])
    println("Numerical calculations give a $(round(100*PWG4,digits=2))% chance of more than 4 total infections")
end

#Use Monte Carlo simulations to determine probability of more than 4 new infections
function Q4c(x0,α;N=1e6)
    WList = []
    TList = []
    for i in 1:N
        xList = [x0]
        yList = [1]
        while true
            x,y = Infected(xList[end],α)
            push!(xList,x)
            push!(yList,y)
            if x*y == 0
                break
            end
        end
        push!(WList,sum(yList)-1)
        push!(TList,length(xList)-1)
    end
    nWG4 = count(W->W>4,WList)
    println("Under Monte Carlo Simulation with $N tests, $(100*nWG4/N)% had more than 4 total infections")
end

#Test code, find probability that the infection runs, should give a value ≈1
function Q4dTest(x0::Int,α::Float64;dz=0.01::Float64)
    zVals = 0.:dz:1.
    #GVal = zero(zVals)
    GVals = []
    Adash = zeros(x0+1)
    Adash[end] = 1#Initial condition

    Pbar = zeros(x0+1,x0+1)
    Q = zeros(x0+1,x0+1)
    for i in 1:x0+1
        Q[i,i] = α^(i-1)
        for j in 1:i-1
            Pbar[i,j] = binomial(i-1,j-1)*(1-α)^(i-j)*α^(j-1)
        end
    end
    E = [1 for _ in 0:x0]
    for z in zVals
        ident = Matrix{Float64}(I,x0+1,x0+1)
        push!(GVals,Adash'*inv(ident-z*Pbar)*z*Q*E)
    end
    PWg4 = 0.0
    for t in 0:x0
        PWg4 += 1/(factorial(t)) * NewtonDiff0(GVals,dz,t)
    end
    println("The total probability is $PWg4")
    plot(zVals,GVals)
    xlabel!("z")
    ylabel!("PGF(z)")
end

#Find probability of more than 4 infections using numerical solutions to an analytical matrix method
function Q4d(x0::Int,α::Float64;dz=0.001::Float64)
    zVals = 0:dz:1
    GVals = zero(zVals)
    A = zeros(x0+1)
    A[end] = 1#Initial condition

    Pbar = zeros(x0+1,x0+1)
    Q = zeros(x0+1,x0+1)
    for i in 1:x0+1
        Q[i,i] = α^(i-1)
        for j in 1:i-1
            Pbar[i,j] = binomial(i-1,j-1)*(1-α)^(i-j)*α^(j-1)
        end
    end
    ident = Matrix{Float64}(I,x0+1,x0+1)
    for j in 1:length(zVals)
        z = zVals[j]
        E = [z^i for i in 0:x0]
        GVals[j] = A'*inv(ident-Pbar)*Q*E
    end
    PWg4 = 0.0
    for x in 0:x0-5
        PWg4 += 1/(factorial(x)) * NewtonDiff0(GVals,dz,x)
    end
    println("The probability that W>=4 is $PWg4")
    plot(zVals,GVals,legend=:none)
    xlabel!("z")
    ylabel!("PGF(z)")
end

# This one's not important
function Q4dW(x0::Int,α::Float64;dz=0.001::Float64)
    zVals = 0:dz:1
    #GVal = zero(zVals)
    GVals = []
    Adash = zeros(x0+1)
    Adash[end] = 1#Initial condition

    Pbar = zeros(x0+1,x0+1)
    Q = zeros(x0+1,x0+1)
    for i in 1:x0+1
        Q[i,i] = α^(i-1)
        for j in 1:i-1
            Pbar[i,j] = binomial(i-1,j-1)*(1-α)^(i-j)*α^(j-1)
        end
    end
    for z in zVals
        E = [z^(x0-i) for i in 0:x0]
        ident = Matrix{Float64}(I,x0+1,x0+1)
        push!(GVals,Adash'*inv(ident-Pbar)*Q*E)
    end
    PWg4 = 0.0
    for w in 5:x0
        PWg4 += 1/(factorial(w)) * NewtonDiff0(GVals,dz,w)
    end
    println("The probability that W>=4 is $PWg4")
    plot(zVals,GVals,legend=:none)
    xlabel!("z")
    ylabel!("PGF(z)")
end

#Newtons formula for numerical derivatives
function NewtonDiff0(xList,dt::Float64,n::Int)
    return 1/(dt^n)*sum([(-1)^(k+n)*binomial(n,k)*xList[1+k] for k in 0:n])
end
## Question 5 - Numerics of Reed-Frost model

#Heatmap of transition probability matrix of Reed-Frost model
function Q5c(x0::Int,α::Float64;T=5::Int)
    P = zeros((x0+1)^2,(x0+1)^2)
    States = Dict(i => ((i-1)÷(x0+1),(i-1)%(x0+1)) for i in 1:(x0+1)^2) #get table of conversions between matrix index and state
    for i in 1:(x0+1)^2
        for j in 1:(x0+1)^2
            xnow,ynow = States[i]
            xnext,ynext = States[j]
            if xnext + ynext == xnow
                P[i,j] = binomial(xnow,xnext)*α^(ynow*xnext)*(1-α^ynow)^ynext
            end
        end
    end
    p = heatmap(P)
    #title!("Heatmap of Reed-Frost Transition Matrix, x_0=$x0,α=$α")
    ylabel!("State 'from'")
    xlabel!("State 'to'")
    display(p)
    return P
end

#Monte Carlo simulation of Reed-Frost model to determine probability of more than 4 total new infections
function Q5d(x0::Int,α::Float64;N=1e6::Float64)
    WList = []
    TList = []
    for i in 1:N
        xList = [x0]
        yList = [1]
        while true
            x,y = Infected(xList[end],α^yList[end])
            push!(xList,x)
            push!(yList,y)
            if y == 0
                break
            end
        end
        push!(WList,sum(yList)-1)
        push!(TList,length(xList)-1)
    end
    nWG4 = count(W->W>4,WList)
    println("Under Monte Carlo Simulation with $N tests, $(100*nWG4/N)% had more than 4 total infections")
end


## Question 6 - Motel
#For N people each with a chance α of avoiding infection, return tuple of (uninfected,infected)
function Infected(N::Int,α::Float64)
    x = sum(rand(N).<α)
    y = N-x
    return x::Int,y::Int
end

# Runs a generation of arrivals to the motel, accepting x0 arrivals each with a chance η of already being infected, within the motel there is a chance p of coming into contact with each person, and β of that contact being sufficient to communicate the infection
function Motel(x0::Int;η=0.05::Float64,p=0.1::Float64,β=0.05::Float64)
    x1,y1 = Infected(x0,1-η)
    xList = [x1]
    yList = [y1]
    while true
        if (yList[end]==0)
            return (xList,yList)
        end
        #Reed-Frost
        α = 1-p*β
        (x,y) = Infected(xList[end],α^(yList[end]))
        push!(xList,x)
        push!(yList,y)
    end
end

#Run many instances of motel generations
function Q6(x0;N=1e6,η=0.05,p=0.1,β=0.05,Loud=true)
    avgList = []
    sumList = []
    bigyList =[]
    for i = 1:N
        xList,yList = Motel(x0,η=η,p=p,β=β)
        #push!(avgList,sum(yList)/max(1,length(yList)-1))
        push!(avgList,Statistics.mean(yList))
        push!(sumList,sum(yList)/x0)
        push!(bigyList,yList...)
    end
    Y = Statistics.mean(avgList)
    Y2 = Statistics.mean(bigyList)
    Z = Statistics.mean(sumList)
    if Loud
        println("The expected rate of infections from $N trials is $Z per person, as opposed to a raw rate of η=$η. We expect $Y2 infectives to leave the motel per day" )
    end
    return Y2,Z
end

#run Q6(.) for provided values of x0 and plot results
function Q6graph(x0List::Array;N=1e6,η=0.05,p=0.1,β=0.05)
    YList = []
    ZList = []
    for x0 in x0List
        Y,Z = Q6(x0;N=1e6,η=0.05,p=0.1,β=0.05,Loud=false)
        push!(YList,Y)
        push!(ZList,Z)
    end
    display(plot(x0List,[YList,ZList],layout=(1,2),title=["Daily infected from motel" "Rate of infection"],xlabel = ["x_0" "x_0"],label=:none))
    return YList, ZList
end

function Q6matrix(x0;η=0.05,p=0.1,β=0.05)
    α = 1-p*β
    P = zeros((x0+1)^2,(x0+1)^2)
    Pbar = zeros((x0+1)^2,(x0+1)^2)
    Q = zeros((x0+1)^2,(x0+1)^2)
    States = Dict(i => ((i-1)÷(x0+1),(i-1)%(x0+1)) for i in 1:(x0+1)^2) #get table of conversions between matrix index and state
    for i in 1:(x0+1)^2
        for j in 1:(x0+1)^2
            xnow,ynow = States[i]
            xnext,ynext = States[j]
            if (xnext + ynext == xnow)
                P[i,j] = binomial(xnow,xnext)*α^(ynow*xnext)*(1-α^ynow)^ynext
                if xnow == xnext
                    Q[i,j] = binomial(xnow,xnext)*α^(ynow*xnext)*(1-α^ynow)^ynext
                else
                    Pbar[i,j] = binomial(xnow,xnext)*α^(ynow*xnext)*(1-α^ynow)^ynext
                end
            end
        end
    end
    A = zeros((x0+1)^2)
    for i in 1:(x0+1)^2
        (x,y) = States[i]
        if ((x + y) == x0)
            A[i] = binomial(x0,x)*((1-η)^x)*(η^y)
        end
    end

    E = [States[(x0+1)^2+1-i][1] for i in 1:(x0+1)^2]
    println(E)
    #rates = [1/t* A'*Pbar^(t-1)*Q*E for t in 1:x0+1]
    EW = sum([A'*Pbar^(t-1)*Q*E for t in 1:x0+1])
    ET = sum([t*A'Pbar^(t-1)*Q*[1 for _ in 1:(x0+1)^2] for t in 1:x0+1])
    return EW/ET
    #return sum([1/t* A'*Pbar^(t-1)*Q*E*(A'Pbar^(t-1)*Q*[1 for _ in 1:(x0+1)^2]) for t in 1:x0+1])
    #
    #return sum([A'*(P^(t-1))*Q*E/t for t in 1:x0+1])
end

function ComparePiecewise(A,B)
    if !(size(A) == size(B))
        println("dimension mismatch")
        return
    end
    for i in 1:size(A)[1]
        for j in size(A)[2]
            if norm(A[i,j] - B[i,j]) > 0.01
                return false
            end
        end
    end
    return true
end
