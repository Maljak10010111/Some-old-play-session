from adv_lib.attacks import projected_gradient_descent as pgd
from adv_lib.attacks import fast_minimum_norm as fmnd
from adv_lib.attacks import auto_pgd as apgd
from robustbench.data import load_cifar10
from robustbench.utils import load_model
from foolbox.attacks import fast_gradient_method as fgm
import torch
import numpy as np

if __name__ == '__main__':
    x_test, y_test = load_cifar10(n_examples=50)

    # models from RobustBench
    model_carmon = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')
    model_standard = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
    model_rice = load_model(model_name='Rice2020Overfitting', dataset='cifar10', threat_model='Linf')

    # # # Adversarial Library - PGD
    advs_carmon= pgd.pgd_linf(model_carmon, x_test, y_test, 0.01)
    advs_std = pgd.pgd_linf(model_standard, x_test, y_test, 0.01)
    advs_rice = pgd.pgd_linf(model_rice, x_test, y_test, 0.03)



    # .eval converts model to evaluation mode
    model_carmon.eval()
    model_standard.eval()
    model_rice.eval()


    with torch.no_grad():
        clean_predictions_carmon = model_carmon(x_test)
        clean_predictions_std = model_standard(x_test)
        clean_predictions_rice = model_rice(x_test)
        # PGD
        advs_predictions_carmon = model_carmon(advs_carmon)
        advs_predictions_std = model_standard(advs_std)
        advs_predictions_rice = model_rice(advs_rice)


    # Convert predictions to class labels
    # CLEAN LABELS
    predicted_labels_clean_carmon = torch.argmax(clean_predictions_carmon, dim=1)
    predicted_labels_clean_std = torch.argmax(clean_predictions_std, dim=1)
    predicted_labels_clean_rice = torch.argmax(clean_predictions_rice, dim=1)
    # ADVERSARIAL LABELS
    predicted_labels_carmon = torch.argmax(advs_predictions_carmon, dim=1)
    predicted_labels_std = torch.argmax(advs_predictions_std, dim=1)
    predicted_labels_rice = torch.argmax(advs_predictions_rice, dim=1)

    # The computation of the accuracy on unpertubated testset and on pertubated one
    accuracy_clean_carmon = torch.mean((predicted_labels_clean_carmon == y_test).float())
    accuracy_clean_std = torch.mean((predicted_labels_clean_std == y_test).float())
    accuracy_clean_rice = torch.mean((predicted_labels_clean_rice == y_test).float())
    accuracy_carmon = torch.mean((predicted_labels_carmon == y_test).float())
    accuracy_standard = torch.mean((predicted_labels_std == y_test).float())
    accuracy_rice = torch.mean((predicted_labels_rice == y_test).float())

    print("Accuracy on clean Carmon test set:", accuracy_clean_carmon.item())
    print("Accuracy on perturbed Carmon test set:", accuracy_carmon.item())
    print("Accuracy on clean Standard test set:", accuracy_clean_std.item())
    print("Accuracy on perturbed Standard test set:", accuracy_standard.item())
    print("Accuracy on clean Rice test set:", accuracy_clean_rice.item())
    print("Accuracy on perturbed Rice test set:", accuracy_rice.item())



    xt_test_fmn, y_test_fmn = load_cifar10(n_examples=40)

    model_carmon_for_fmn = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')
    model_engstrom_for_fmn = load_model(model_name='Engstrom2019Robustness', dataset='cifar10', threat_model='Linf')
    model_rice_for_fmn = load_model(model_name='Rice2020Overfitting', dataset='cifar10', threat_model='Linf')

    advs_fmn = fmnd.fmn(model_carmon_for_fmn,xt_test_fmn,y_test_fmn,np.inf)
    advs_fmn = fmnd.fmn(model_engstrom_for_fmn, xt_test_fmn, y_test_fmn, np.inf)
    advs_fmn = fmnd.fmn(model_rice_for_fmn, xt_test_fmn, y_test_fmn, np.inf)

    model_carmon_for_fmn.eval()
    model_engstrom_for_fmn.eval()
    model_rice_for_fmn.eval()

    with torch.no_grad():
        clean_predictions_carmon = model_carmon_for_fmn(xt_test_fmn)
        advs_predictions_carmon_fmn = model_carmon_for_fmn(advs_fmn)
        clean_predictions_engstrom = model_engstrom_for_fmn(xt_test_fmn)
        advs_predictions_engstrom_fmn = model_engstrom_for_fmn(advs_fmn)
        clean_predictions_rice = model_rice_for_fmn(xt_test_fmn)
        advs_predictions_rice_fmn = model_rice_for_fmn(advs_fmn)


    predicted_labels_clean_carmon = torch.argmax(clean_predictions_engstrom, dim=1)
    predicted_labels_carmon_fmn = torch.argmax(advs_predictions_engstrom_fmn, dim=1)
    predicted_labels_clean_engstrom = torch.argmax(clean_predictions_engstrom, dim=1)
    predicted_labels_engstrom_fmn = torch.argmax(advs_predictions_engstrom_fmn, dim=1)
    predicted_labels_clean_rice = torch.argmax(clean_predictions_rice, dim=1)
    predicted_labels_rice_fmn = torch.argmax(advs_predictions_rice_fmn, dim=1)


    accuracy_clean = torch.mean((predicted_labels_clean_rice == y_test_fmn).float())
    accuracy_rice_fmn = torch.mean((predicted_labels_rice_fmn == y_test_fmn).float())

    print("Accuracy on clean Rice test set:", accuracy_clean.item())
    print("Accuracy on perturbed Rice test set with Linf fmn attack:", accuracy_rice_fmn.item())
    """
    Accuracy on clean Carmon test set: 0.8999999761581421
    Accuracy on perturbed Carmon2019Unlabeled test set with Linf fmn attack: 0.05000000074505806
    """
    """
    Accuracy on clean Engstrom test set: 0.875
    Accuracy on perturbed Engstrom2019 test set with Linf fmn attack: 0.0
    """
    """
    Accuracy on clean Rice test set: 0.8500000238418579
    Accuracy on perturbed Rice test set with Linf fmn attack: 0.0
    """


    xt_test_apgd, y_test_apgd = load_cifar10(n_examples=30)
    model_carmon_for_autopgd = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')

    advs_apgd = apgd.apgd(model_carmon_for_autopgd, xt_test_apgd, y_test_apgd, 0.03, np.inf)

    model_carmon_for_autopgd.eval()

    with torch.no_grad():
        clean_predictions_carmon = model_carmon_for_autopgd(xt_test_apgd)
        advs_predictions_carmon_apgd = model_carmon_for_autopgd(advs_apgd)

    predicted_labels_clean_carmon = torch.argmax(clean_predictions_carmon, dim=1)
    predicted_labels_carmon_apgd = torch.argmax(advs_predictions_carmon_apgd, dim=1)

    accuracy_clean_model = torch.mean((predicted_labels_clean_carmon == y_test_apgd).float())
    accuracy_perturbed_carmon_apgd = torch.mean((predicted_labels_carmon_apgd == y_test_apgd).float())

    print("Accuracy of a clean model:", accuracy_clean_model.item())
    print("Accuracy of a perturbed model:", accuracy_perturbed_carmon_apgd.item())
    # """
    # eps = 0.02
    # Accuracy of a clean model: 0.9333333373069763
    # Accuracy of a perturbed model: 0.7666666507720947
    # """

    # """
    # epsilon = 0.03
    # Accuracy of a clean model: 0.9333333373069763
    # Accuracy of a perturbed model: 0.6333333253860474
    # """

    # Fast Gradient Sign from Foolbox
    xt_test_apgd, y_test_apgd = load_cifar10(n_examples=30)
    model_carmon_for_fgs = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')

    advs_fgs_foolbox = fgm.LinfFastGradientAttack.run(model_carmon_for_fgs, xt_test_apgd, xt_test_apgd, criterion='Misclassification')




