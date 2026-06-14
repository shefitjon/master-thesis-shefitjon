<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Console\Command;

class AbReport extends \Symfony\Component\Console\Command\Command
{
    public function __construct(
        private readonly \Bregu\CartRecoveryAb\Model\EventLogger $eventLogger,
        private readonly \Bregu\CartRecoveryAb\Model\ChiSquareTest $chiSquareTest,
        ?string $name = null
    ) {
        parent::__construct($name);
    }

    protected function configure(): void
    {
        $this->setName('bregu:ab:report');
        $this->setDescription('Report cart-recovery A/B results with a chi-square significance test.');
    }

    protected function execute(
        \Symfony\Component\Console\Input\InputInterface $input,
        \Symfony\Component\Console\Output\OutputInterface $output
    ): int {
        $buckets = [
            \Bregu\CartRecoveryAb\Model\AbAssigner::BUCKET_CONTROL => $this->emptyRow(),
            \Bregu\CartRecoveryAb\Model\AbAssigner::BUCKET_TREATMENT => $this->emptyRow(),
        ];

        foreach ($this->eventLogger->aggregateByBucket() as $row) {
            $name = (string) $row['bucket'];
            if (isset($buckets[$name])) {
                $buckets[$name] = $row;
            }
        }

        foreach ($buckets as $name => $row) {
            $exposures = (int) $row['exposures'];
            $conversions = (int) $row['conversions'];
            $rate = $exposures > 0 ? ($conversions / $exposures) * 100 : 0.0;
            $revenue = (float) $row['revenue'];
            $output->writeln(sprintf(
                '<info>%-10s</info> exposures=%d impressions=%d conversions=%d rate=%.2f%% revenue=%.2f',
                $name,
                $exposures,
                (int) $row['impressions'],
                $conversions,
                $rate,
                $revenue
            ));
        }

        $control = $buckets[\Bregu\CartRecoveryAb\Model\AbAssigner::BUCKET_CONTROL];
        $treatment = $buckets[\Bregu\CartRecoveryAb\Model\AbAssigner::BUCKET_TREATMENT];
        $test = $this->chiSquareTest->test(
            (int) $control['conversions'],
            (int) $control['exposures'],
            (int) $treatment['conversions'],
            (int) $treatment['exposures']
        );

        $output->writeln('');
        $output->writeln(sprintf(
            'chi-square = %.4f, p = %.4f%s',
            $test['chi2'],
            $test['p'],
            $test['p'] < 0.05 ? '  (significant at 0.05)' : '  (not significant at 0.05)'
        ));

        return \Symfony\Component\Console\Command\Command::SUCCESS;
    }

    /**
     * @return array<string, mixed>
     */
    private function emptyRow(): array
    {
        return ['exposures' => 0, 'impressions' => 0, 'conversions' => 0, 'revenue' => 0.0];
    }
}
