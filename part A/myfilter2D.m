%Function to implement filter's frequency respone H(u,v)
function H=myfilter2D(type,M,N,D0,n) %myfilter2D(type,row,col,radius,order)

P=floor(M/2); Q=floor(N/2); %Getting centre of image
D=zeros(M,N); H=zeros(M,N);

switch type
    case 'idealLPF'
        for u=1:M
            for v=1:N
            D(u,v)=sqrt((u-P)^2+(v-Q)^2); %Distance from image centre
            if D(u,v)<=D0
                H(u,v)=1;
            end
            end
        end
            
    case 'butterLPF'
        for u=1:M
            for v=1:N
                D(u,v)=sqrt((u-P)^2+(v-Q)^2);
                H(u,v)=1/(1+(D(u,v)/D0)^(2*n));
            end
        end
    case 'gaussianLPF'
        for u=1:M
            for v=1:N
                D(u,v)=sqrt((u-P)^2+(v-Q)^2);
                H(u,v)=exp(-((D(u,v)^2)/(2*D0^2)));
            end
        end
end
 
            