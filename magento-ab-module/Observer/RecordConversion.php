<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Observer;

class RecordConversion implements \Magento\Framework\Event\ObserverInterface
{
    public function __construct(
        private readonly \Bregu\CartRecoveryAb\Model\Config $config,
        private readonly \Bregu\CartRecoveryAb\Model\EventLogger $eventLogger
    ) {
    }

    public function execute(\Magento\Framework\Event\Observer $observer): void
    {
        if (!$this->config->isEnabled()) {
            return;
        }

        $order = $observer->getEvent()->getData('order');
        if ($order === null) {
            $orders = $observer->getEvent()->getData('orders');
            $order = is_array($orders) ? reset($orders) : null;
        }
        if ($order === null) {
            return;
        }

        $quoteId = (int) $order->getQuoteId();
        if ($quoteId <= 0) {
            return;
        }

        $this->eventLogger->recordConversion(
            (string) $quoteId,
            (int) $order->getId(),
            (float) $order->getGrandTotal()
        );
    }
}
