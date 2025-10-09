function plotpanelsStokes(panels,f)

clf
for ii=1:size(panels,3)
  p=panels(2:end,:,ii);
  n=panels(1,1,ii);
  patch(p(1:n,1),p(1:n,2),p(1:n,3),f(ii)*ones(n,1))
end


