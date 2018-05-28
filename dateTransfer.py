def dateTransfer():
	def transferdate(start,end):
	weekend=[]
    
    for i in range(start,end+1):
        print(i)
        if (i-start)%7==0 or (i-start)%7==1:
            weekend.append(i)
    retur​n weekend


	ss_9=transferdate(20170902,20170930)
	ss_10=transferdate(20171007,20171031)


	ss_11=transferdate(20171104,20171130)

	ss_12=transferdate(20171202,20171231)

	ss_1=transferdate(20180106,20180131)
	holiday=[20171001,20171002,20171003,20171004,20171005,20171006,20171111,20180203,20180204]
	return ss_9+ss_1+ss_12+ss_11+holiday+ss_10