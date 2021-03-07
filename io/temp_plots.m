function temp_plots()

    %% BARS
%     f = figure;
%     hold on;
%     ad = 53.86; ad_da = 63.66; ad_sota = 55.7; ad_sota_da = 57.1;
%     aw = 47.59; aw_da = 63.55; aw_sota = 50.6; aw_sota_da = 53.1;
%     da = 49.18; da_da = 58.41; da_sota = 46.5; da_sota_da = 51.1;
%     dw = 92.33; dw_da = 92.80; dw_sota = 93.1; dw_sota_da = 94.6;
%     wa = 44.68; wa_da = 58.41; wa_sota = 43.0; wa_sota_da = 47.3;
%     wd = 93.33; wd_da = 93.87; wd_sota = 97.4; wd_sota_da = 98.2;
%     normALL = [ad; aw; da; dw; wa; wd];
%     daALL = [ad_da; aw_da; da_da; dw_da; wa_da; wd_da];
%     bar(daALL, 0.4, 'FaceColor',[0 0.7 0.7]);
%     bar(normALL);
%     normSotaALL = [ad_sota; aw_sota; da_sota; dw_sota; wa_sota; wd_sota];
%     daSotaALL = [ad_sota_da; aw_sota_da; da_sota_da; dw_sota_da; wa_sota_da; wd_sota_da];
%     bar(daSotaALL, 0.15, 'FaceColor',[0.85 0.1 0.1]);
%     bar(normSotaALL, 0.3, 'FaceColor',[0.75 0.3 0.3]);
%     hold off;
%     
%     title('Unsupervised Classification - Office Dataset (CNN Features - fc7)', 'FontSize', 14);
%     set(gca,'XTickLabel',{'A > D',' A > W',' D > A',' D > W',' W > A',' W > D'});
%     set(f, 'Position', [300, 150, 1200, 400]);
%     axis([0.5,6.5,0,100]);
%     d = 0;
%     d_f = 0;
%     pos = -9;
%     pos_f = -4;
%     legend({'Ours with DA', 'Ours w/o Adaptation', 'Sun et al. "16 with DA', 'Sun et al. "16 w/o DA'}, 'Location', 'northwest');

    %% Lines
%     figure; title('[WEBCAM > DSLR] K-Means vs SVM clustering 31<>31 (corresp + acc)', 'FontSize', 12);
%     kMeansCP = [40 43 48 48 48];
%     kMeansCR = [100 100 100 100 100];
%     kMeansAc = [62.73 63.02 57.49 92.66 57.54 90.79];
%     SVM_CP = [90 91];
%     SVM_CR = [100 100 100 100 100];
%     SVM_Ac = [65.56 64.01 57.48 93.48 57.27 ];
%     hold on;
%     % Kmeans
%     plot(1:5,kMeansCP,'-s', 'Color',[0,0.5,0.7], 'LineWidth', 2, 'MarkerSize', 5, 'LineSmoothing', 'on');
%     text(1, kMeansCP(1), '   kMeans corresp. precision');
%     plot(1:5,kMeansCR,'-s', 'Color',[0,0.9,0.7], 'LineWidth', 2, 'MarkerSize', 5, 'LineSmoothing', 'on');
%     text(1, kMeansCR(1), '   kMeans corresp. recall');
%     plot(5,kMeansAc(5),'x', 'Color', [0,0.5,0.7], 'LineWidth', 2, 'MarkerSize', 25);
%     % SVM
%     plot(1:5,SVM_CP,'-s', 'Color',[0.5,0,0.7], 'LineWidth', 2, 'MarkerSize', 5, 'LineSmoothing', 'on');
%     text(1, SVM_CP(1), '   SVM corresp. precision');
%     plot(1:5,SVM_CR,'-s', 'Color',[0.9,0,0.7], 'LineWidth', 2, 'MarkerSize', 5, 'LineSmoothing', 'on');
%     text(1, SVM_CR(1), '   SVM corresp. recall');
%     plot(5,SVM_Ac(5),'x', 'Color', [0.5,0,0.7], 'LineWidth', 2, 'MarkerSize', 25);
%     axis([1 5 0 100]);
%     xlabel('number iterations');
%     ylabel('accuracy');
%     hold off;

    %% Lines 2
    figure; title('[DSLR - Webcam (small domain shift)] Lambda impact in 31 (src) <> Instance (tgt) correspodences', 'FontSize', 12);
%     lambdaPmedian = [46 45 46 50 53 53 53 54];
%     lambdaRmedian = [74 74 74 82 96 98 99 99];
%     lambdaAmedian = [54.93 55.04 55.12 55.84 57.72 57.99 57.66 57.65];
    lambdaP_AW = [43 45 54 54];
    lambdaR_AW = [7 54 100 100];
    lambdaP_DW = [100 98 89 89 79 79];
    lambdaR_DW = [0 12 66 66 100 100];
%     lambdaA = [60.17 59.82 60.51 61.87 61.52 61.67 61.67 61.67];
    hold on;
    % Kmeans
%     xRangeAW = [0.4 0.5 0.75 1.0];
%     plot(xRangeAW,lambdaP_AW,'-s', 'Color',[0,0.5,0.7], 'LineWidth', 2, 'MarkerSize', 5, 'LineSmoothing', 'on');
%     text(0.3, lambdaP_AW(1)-2, '   corresp. precision');
%     plot(xRangeAW,lambdaR_AW,'-s', 'Color',[1.0,0.0,0.7], 'LineWidth', 2, 'MarkerSize', 5, 'LineSmoothing', 'on');
%     text(0.3, lambdaR_AW(1)-2, '   corresp. recall');
    xRangeDW = [0.3 0.4 0.5 0.6 0.8 1.0];
    plot(xRangeDW,lambdaP_DW,'-s', 'Color',[0,0.5,0.7], 'LineWidth', 2, 'MarkerSize', 5, 'LineSmoothing', 'on');
%     text(0.3, lambdaP_DW(1)-2, '   corresp. precision');
    plot(xRangeDW,lambdaR_DW,'-s', 'Color',[1.0,0.0,0.7], 'LineWidth', 2, 'MarkerSize', 5, 'LineSmoothing', 'on');
%     text(0.3, lambdaR_DW(1)-2, '   corresp. recall');
%     plot(0.3:0.1:1.0,lambdaA,'-x', 'Color', [0,0.5,0.7], 'LineWidth', 2, 'MarkerSize', 25, 'LineSmoothing', 'on');
%     text(0.3, lambdaA(1)-2, '   acc. classification');
%     plot(0.3:0.1:1.0,lambdaPmedian,'-s', 'Color',[0.5,0,0.7], 'LineWidth', 2, 'MarkerSize', 5, 'LineSmoothing', 'on');
%     text(0.3, lambdaPmedian(1)-2, '   corresp. precision');
%     plot(0.3:0.1:1.0,lambdaRmedian,'-s', 'Color',[0.9,0,0.7], 'LineWidth', 2, 'MarkerSize', 5, 'LineSmoothing', 'on');
%     text(0.3, lambdaRmedian(1)-2, '   corresp. recall');
%     plot(0.3:0.1:1.0,lambdaAmedian,'-x', 'Color', [0.5,0,0.7], 'LineWidth', 2, 'MarkerSize', 25, 'LineSmoothing', 'on');
%     text(0.3, lambdaAmedian(1)-2, '   acc. classification');
    legend('precision','recall');
    axis([0.3 1 0 100]);
    xlabel('ignore correspondences (lambda distance value w.r.t max(distance))');
    ylabel('accuracy');
    hold off;    
end

