function outimg = integralImg (inimg)
    outimg = cumsum(cumsum(double(inimg),2));
end
