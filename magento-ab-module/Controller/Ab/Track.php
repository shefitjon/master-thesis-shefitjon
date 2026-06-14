<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Controller\Ab;

class Track implements
    \Magento\Framework\App\Action\HttpPostActionInterface,
    \Magento\Framework\App\CsrfAwareActionInterface
{
    public function __construct(
        private readonly \Magento\Framework\App\RequestInterface $request,
        private readonly \Magento\Framework\Controller\Result\JsonFactory $jsonFactory,
        private readonly \Magento\Framework\Data\Form\FormKey\Validator $formKeyValidator,
        private readonly \Bregu\CartRecoveryAb\Model\Config $config,
        private readonly \Bregu\CartRecoveryAb\Model\EventLogger $eventLogger
    ) {
    }

    public function createCsrfValidationException(
        \Magento\Framework\App\RequestInterface $request
    ): ?\Magento\Framework\App\Request\InvalidRequestException {
        $response = $this->jsonFactory->create()->setData(['ok' => false]);
        return new \Magento\Framework\App\Request\InvalidRequestException($response);
    }

    public function validateForCsrf(\Magento\Framework\App\RequestInterface $request): ?bool
    {
        return $this->formKeyValidator->validate($request);
    }

    public function execute()
    {
        $result = $this->jsonFactory->create();
        if (!$this->config->isEnabled()) {
            return $result->setData(['ok' => false]);
        }

        $sessionKey = trim((string) $this->request->getParam('session_key'));
        $event = (string) $this->request->getParam('event');
        if ($sessionKey === '') {
            return $result->setData(['ok' => false]);
        }

        switch ($event) {
            case 'exposure':
                $this->eventLogger->recordExposure(
                    $sessionKey,
                    (string) $this->request->getParam('bucket'),
                    (float) $this->request->getParam('risk'),
                    (string) $this->request->getParam('signal')
                );
                break;
            case 'impression':
                $this->eventLogger->markShown($sessionKey);
                break;
            case 'dismiss':
                $this->eventLogger->markDismissed($sessionKey);
                break;
            default:
                return $result->setData(['ok' => false]);
        }

        return $result->setData(['ok' => true]);
    }
}
