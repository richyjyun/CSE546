% A9 figures
% a)
x = linspace(-5,5,100);
y = (x-2)/2;
figure; plot(x,y)
set(gca,'xaxislocation','origin')
set(gca,'yaxislocation','origin')
box off
xlim([-4,4])
ylim([-4,4])
xlabel('x_1')
ylabel('x_2')

% b)
x = linspace(-5,5,100);
y = linspace(-5,5,100);
[X,Y] = meshgrid(x,y);
Z = 2-X-Y;
figure; surf(X,Y,Z);
shading interp
set(gca,'xaxislocation','origin')
set(gca,'yaxislocation','origin')
xlim([-4,4]); ylim([-4,4]); zlim([-4,4])
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');