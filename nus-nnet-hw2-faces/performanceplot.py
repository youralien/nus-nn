import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def performanceplot(cost_record, train_error_record, test_error_record, fig_outfile):
    plt.subplot(2,1,1)
    plt.plot(cost_record)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.subplot(2,1,2)
    plt.hold('on')
    plt.plot(train_error_record, 'b')
    plt.plot(test_error_record, 'r')
    plt.xlabel('epoch')
    plt.ylabel('error')

    plt.tight_layout()
    plt.savefig(fig_outfile, bbox_inches='tight')

