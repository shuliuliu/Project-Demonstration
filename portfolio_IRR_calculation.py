TOLERANCE = 0.00001
# %%
def readFile(path):
    with open(path, "rt") as f:
        return f.read()

def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

# %%
# bisection method
def findNPV(cashflow, delta, r):
  if 1.0 + r == 0.0:
    return None
  NPV = 0
  for i in range(len(cashflow)):
    den = ((1.0 + r + 0j) ** (delta[i] / 365.0)).real
    NPV += cashflow[i] / den
  return NPV

def RootFinding_bisection_method(cashflow, delta, low, high):
  stepSize = 0.001
  numRun = int((high - low) / stepSize + 1)
  for i in range(numRun):
    lowbound = low + i * stepSize
    highbound = lowbound + stepSize

    if (lowbound + 1 == 0.0 or highbound + 1 == 0.0):
      continue

    npvLow = findNPV(cashflow, delta, lowbound)
    npvHigh = findNPV(cashflow, delta, highbound)

    if (npvLow * npvHigh > 0):
      continue

    maxItr = 512
    itr = 0

    while (itr < maxItr):
      itr += 1
      mid = (lowbound + highbound) * 0.5

      if mid + 1 == 0.0:
        mid += (stepSize / 100)

      npvMid = findNPV(cashflow, delta, mid)

      if (abs(npvMid - 0) < TOLERANCE):
        return mid

      if (npvMid * npvLow > 0):
        lowbound = mid
      else:
        highbound = mid
    return None

# %%
# Newton's method
def RootFinding_newtons_method(cashflow, delta):
  positive = False
  negative = False

  for i in range(len(cashflow)):
    if (cashflow[i] > 0):
      positive = True
    if (cashflow[i] < 0):
      negative = True

  if (not positive or not negative):
    return None

  guess = 0.1
  resultRate = guess

  epsMax = 1e-4
  iterMax = 100000

  #Implement Newton's method
  iteration = 0
  contLoop = True

  while (contLoop and (++iteration < iterMax)):
    resultValue = 0
    irrResultDeriv = 0
    for i in range(len(cashflow)):
      resultValue  += cashflow[i] / pow((1+resultRate), delta[i] / 365)

    for j in range(1,len(cashflow)):
      frac = cashflow[j] / 365
      irrResultDeriv -= frac * cashflow[j] / pow((1+resultRate), (frac + 1))

    newRate = resultRate - resultValue / irrResultDeriv
    epsRate = abs(newRate - resultRate)
    resultRate = newRate
    contLoop = (epsRate > epsMax) and (abs(resultValue) > epsMax)

  if (contLoop):
    return None

  return resultRate

# %%
cashflow = []
delta    = []

for line in readFile("testData.csv").splitlines():
  currData = line.split(",")
  cashflow.append(float(currData[4]))
  delta.append(float(currData[3]))

print(RootFinding_bisection_method(cashflow, delta, -1, 5))
print(RootFinding_newtons_method(cashflow, delta))



# %%
"""
select
   id,
   scenario_date,
   cash_flow_date,
   date_diff(cash_flow_date, min(cash_flow_date) over (partition by id), day) as date_delta,
   sum(cash_flow) as cash_flow
 from (
   select 
       cash.id,
       scenario_date,
       cash_flow_date,
       cash_flow_amount as cash_flow
   from `cash_flow_fact` cash
   full join `position_fact` position
   on cash.id = position.id
   -- need to exclude cash_flow_type = Settled Proceeds for both trs and non-trs positions
   where (cash_flow_type != 'Settled Proceeds' and trs_id is not null)
   or (trs_id is null and cash_flow_type != "Notional Proceeds")
   and cash_flow_date <= scenario_date
   UNION ALL
   select 
       id,
       scenario_date,
       cash_flow_date,
       coalesce((case when trs_id is not null then net_leverage else book_dirty_nmv end),0) as cash_flow
   from `position_fact`
   -- only include security position type
   where number_of_trades > 0
 )
 where scenario_date = '2018-08-02'
 and id = 'aaa'
 group by 1,2,3
 order by 1,2,3
"""


# %%
# Example data in the testData.csv file
#######################################
# #aaa	8/2/18	3/28/17	0	-100
# #aaa	8/2/18	5/29/17	62	-200
# #aaa	8/2/18	9/30/17	186	-300
# #aaa	8/2/18	2/26/18	335	250
# #aaa	8/2/18	3/5/18	342	-400
# #aaa	8/2/18	4/2/18	370	-150
# #aaa	8/2/18	8/2/18	492	950