<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Observer;

class FreezeOnCartAdd implements \Magento\Framework\Event\ObserverInterface
{
    public function __construct(
        private readonly \Bregu\CartRecoveryAb\Model\Config $config,
        private readonly \Bregu\CartRecoveryAb\Model\SessionSignals $signals,
        private readonly \Magento\Framework\Stdlib\DateTime\DateTime $date
    ) {
    }

    public function execute(\Magento\Framework\Event\Observer $observer): void
    {
        if (!$this->config->isEnabled()) {
            return;
        }

        $this->signals->freezeAtCartAdd($this->date->gmtTimestamp());
    }
}
