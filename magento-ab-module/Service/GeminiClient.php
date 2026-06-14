<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Service;

class GeminiClient
{
    private const ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent';
    private const MODEL = 'gemini-2.5-flash-lite';
    private const TIMEOUT_SECONDS = 10;

    public function __construct(
        private readonly \Magento\Framework\HTTP\Client\Curl $curl,
        private readonly \Bregu\CartRecoveryAb\Model\Config $config,
        private readonly \Psr\Log\LoggerInterface $logger
    ) {
    }

    public function generate(string $prompt): ?string
    {
        $key = $this->config->getGeminiApiKey();
        if ($key === '') {
            return null;
        }

        $url = sprintf(self::ENDPOINT, self::MODEL);
        $payload = json_encode([
            'contents' => [['parts' => [['text' => $prompt]]]],
            'generationConfig' => [
                'temperature' => 0.7,
                'maxOutputTokens' => 80,
                'thinkingConfig' => ['thinkingBudget' => 0],
            ],
        ]);

        try {
            $this->curl->setTimeout(self::TIMEOUT_SECONDS);
            $this->curl->addHeader('Content-Type', 'application/json');
            $this->curl->addHeader('x-goog-api-key', $key);
            $this->curl->post($url, (string) $payload);

            if ($this->curl->getStatus() !== 200) {
                $this->logger->warning('Bregu_CartRecoveryAb: Gemini HTTP ' . $this->curl->getStatus());
                return null;
            }

            $body = json_decode($this->curl->getBody(), true);
            $text = $body['candidates'][0]['content']['parts'][0]['text'] ?? null;

            return $text !== null ? trim((string) $text) : null;
        } catch (\Throwable $e) {
            $this->logger->warning('Bregu_CartRecoveryAb: Gemini call failed — ' . $e->getMessage());
            return null;
        }
    }
}
