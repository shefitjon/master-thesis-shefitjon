<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Observer;

class RegisterModuleForHyvaConfig implements \Magento\Framework\Event\ObserverInterface
{
    public function __construct(
        private readonly \Magento\Framework\Component\ComponentRegistrar $componentRegistrar
    ) {
    }

    public function execute(\Magento\Framework\Event\Observer $observer): void
    {
        $config = $observer->getData('config');
        $path = $this->componentRegistrar->getPath(
            \Magento\Framework\Component\ComponentRegistrar::MODULE,
            'Bregu_CartRecoveryAb'
        );
        if (!$path) {
            return;
        }

        $extensions = (array) $config->getData('extensions');
        $extensions[] = ['src' => substr($path, strlen(BP) + 1)];
        $config->setData('extensions', $extensions);
    }
}
